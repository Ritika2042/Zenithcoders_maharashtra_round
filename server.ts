import express from 'express';
import { createServer as createViteServer } from 'vite';
import { spawn, spawnSync, ChildProcess } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : 3000;

app.use(express.json());

interface ResolvedPython {
  cmd: string;
  argsPrefix: string[];
  useShell: boolean;
  version: string;
}

let resolvedPython: ResolvedPython | null = null;
let pythonProc: ChildProcess | null = null;
let reqIdCounter = 1;
const pendingRequests = new Map<number, { resolve: (val: any) => void; reject: (err: any) => void; timer: NodeJS.Timeout }>();
let isBridgeReady = false;
let bridgeError: string | null = null;
let consecutiveFailures = 0;
const MAX_CONSECUTIVE_FAILURES = 3;

/**
 * Dynamically resolves the Python executable.
 * Prioritizes process.env.PYTHON, then platform-specific candidates ('python', 'py', 'python3').
 * Tests candidate execution with --version.
 */
function resolvePythonExecutable(): ResolvedPython | null {
  if (resolvedPython) return resolvedPython;

  const isWin = process.platform === 'win32';
  const candidates: string[] = [];

  if (process.env.PYTHON) {
    candidates.push(process.env.PYTHON);
  }

  if (isWin) {
    // On Windows, 'python' and 'py' are standard; 'python3' may be an alias or missing
    candidates.push('python', 'py', 'python3');
  } else {
    // On Unix/Linux/macOS
    candidates.push('python3', 'python');
  }

  for (const cmd of candidates) {
    // Try both direct execution and shell execution on Windows
    const shellOptions = isWin ? [false, true] : [false];
    for (const useShell of shellOptions) {
      try {
        const res = spawnSync(cmd, ['--version'], {
          encoding: 'utf-8',
          stdio: 'pipe',
          shell: useShell,
          timeout: 4000,
        });

        if (!res.error && res.status === 0) {
          const versionOutput = (res.stdout || res.stderr || '').trim();
          if (versionOutput.toLowerCase().includes('python')) {
            console.log(`[ReLearn Bridge] Resolved Python via '${cmd}' (${versionOutput}, shell: ${useShell})`);
            resolvedPython = {
              cmd,
              argsPrefix: [],
              useShell,
              version: versionOutput,
            };
            return resolvedPython;
          }
        }
      } catch {
        // Continue trying next candidate
      }
    }
  }

  return null;
}

function initPythonBridge() {
  const py = resolvePythonExecutable();
  if (!py) {
    bridgeError = `No working Python interpreter found (checked: ${process.env.PYTHON ? process.env.PYTHON + ', ' : ''}${process.platform === 'win32' ? 'python, py, python3' : 'python3, python'}). Please ensure Python is installed and added to PATH, or set the PYTHON environment variable.`;
    console.error(`[ReLearn Bridge] ${bridgeError}`);
    return;
  }

  const scriptPath = path.join(__dirname, 'src', 'models', 'relearn_bridge.py');

  try {
    pythonProc = spawn(py.cmd, [...py.argsPrefix, scriptPath], {
      cwd: __dirname,
      stdio: ['pipe', 'pipe', 'pipe'],
      shell: py.useShell,
    });
  } catch (err: any) {
    bridgeError = `Failed to spawn Python process: ${err.message}`;
    console.error(`[ReLearn Bridge] ${bridgeError}`);
    return;
  }

  let buffer = '';

  pythonProc.stdout?.on('data', (data: Buffer) => {
    buffer += data.toString();
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;

      try {
        const msg = JSON.parse(trimmed);
        if (msg.status === 'READY') {
          isBridgeReady = true;
          consecutiveFailures = 0;
          bridgeError = null;
          console.log('[ReLearn Bridge] Python pipeline initialized and READY');
          continue;
        }

        if (msg.id && pendingRequests.has(msg.id)) {
          const { resolve, timer } = pendingRequests.get(msg.id)!;
          clearTimeout(timer);
          pendingRequests.delete(msg.id);
          resolve(msg);
        }
      } catch (err) {
        console.error('[ReLearn Bridge] Failed to parse stdout line:', trimmed, err);
      }
    }
  });

  // Make sure stderr from the Python process is visible for debugging
  pythonProc.stderr?.on('data', (data: Buffer) => {
    const errText = data.toString();
    process.stderr.write(`[ReLearn Bridge stderr] ${errText}`);
  });

  pythonProc.on('error', (err) => {
    console.error('[ReLearn Bridge] Process error:', err);
    isBridgeReady = false;
  });

  pythonProc.on('exit', (code, signal) => {
    console.warn(`[ReLearn Bridge] Process exited with code ${code} signal ${signal}.`);
    isBridgeReady = false;
    pythonProc = null;

    // Reject any pending requests immediately
    for (const [id, req] of pendingRequests.entries()) {
      clearTimeout(req.timer);
      req.reject(new Error(`Python bridge process exited unexpectedly (code ${code})`));
    }
    pendingRequests.clear();

    // Prevent the bridge from repeatedly spawning broken processes if Python fails repeatedly
    consecutiveFailures++;
    if (consecutiveFailures >= MAX_CONSECUTIVE_FAILURES) {
      bridgeError = `Python bridge process failed ${consecutiveFailures} consecutive times (last exit code: ${code}). Halting automatic respawn.`;
      console.error(`[ReLearn Bridge] ${bridgeError}`);
      return;
    }

    const delayMs = Math.min(1000 * Math.pow(2, consecutiveFailures - 1), 5000);
    console.warn(`[ReLearn Bridge] Will attempt restart in ${delayMs}ms (attempt ${consecutiveFailures}/${MAX_CONSECUTIVE_FAILURES})...`);
    setTimeout(initPythonBridge, delayMs);
  });
}

function sendBridgeRequest(payload: any, timeoutMs = 15000): Promise<any> {
  return new Promise((resolve, reject) => {
    if (!pythonProc || !pythonProc.stdin || pythonProc.killed) {
      return reject(new Error('Python bridge process is not running'));
    }

    const id = reqIdCounter++;
    payload.id = id;

    const timer = setTimeout(() => {
      if (pendingRequests.has(id)) {
        pendingRequests.delete(id);
        reject(new Error('Python bridge request timed out'));
      }
    }, timeoutMs);

    pendingRequests.set(id, { resolve, reject, timer });

    try {
      pythonProc.stdin.write(JSON.stringify(payload) + '\n');
    } catch (e) {
      clearTimeout(timer);
      pendingRequests.delete(id);
      reject(e);
    }
  });
}

// Start python bridge
initPythonBridge();

// API Endpoints
app.get('/api/relearn/health', (_req, res) => {
  res.json({
    status: isBridgeReady ? 'ok' : 'initializing',
    bridgeReady: isBridgeReady,
    pythonExecutable: resolvedPython ? resolvedPython.cmd : null,
    pythonVersion: resolvedPython ? resolvedPython.version : null,
    error: bridgeError,
    timestamp: new Date().toISOString()
  });
});

app.post('/api/relearn/diagnose', async (req, res) => {
  try {
    const { question, correct_answer, student_answer, student_reasoning } = req.body;
    console.log('[API /diagnose] Incoming payload:', {
      question: typeof question === 'string' ? question.replace(/\n/g, '\\n') : question,
      correct_answer,
      student_answer,
      student_reasoning
    });
    const response = await sendBridgeRequest({
      action: 'diagnose',
      question: question || '',
      correct_answer: correct_answer || '',
      student_answer: student_answer || '',
      student_reasoning: student_reasoning || ''
    });

    if (response.success) {
      console.log('[API /diagnose] Success. Misconception:', response.data?.diagnosis?.misconception_id);
      res.json(response.data);
    } else {
      console.error('[API /diagnose] Failed:', response.error);
      res.status(500).json({ error: response.error || 'Diagnosis failed' });
    }
  } catch (err: any) {
    console.error('[API /diagnose] Error:', err);
    res.status(500).json({ error: err.message || 'Server error' });
  }
});

app.post('/api/relearn/evaluate', async (req, res) => {
  try {
    const { session, followup_answer, followup_reasoning } = req.body;
    console.log('[API /evaluate] Incoming follow-up submission:', {
      followup_answer,
      followup_reasoning
    });
    const response = await sendBridgeRequest({
      action: 'evaluate',
      session: session || {},
      followup_answer: followup_answer || '',
      followup_reasoning: followup_reasoning || ''
    });

    if (response.success) {
      console.log('[API /evaluate] Success. Resolution:', response.data?.resolution?.status);
      res.json(response.data);
    } else {
      console.error('[API /evaluate] Failed:', response.error);
      res.status(500).json({ error: response.error || 'Evaluation failed' });
    }
  } catch (err: any) {
    console.error('[API /evaluate] Error:', err);
    res.status(500).json({ error: err.message || 'Server error' });
  }
});

// Vite Middleware
async function startServer() {
  const isProd = process.env.NODE_ENV === 'production';

  if (!isProd) {
    const vite = await createViteServer({
      server: { middlewareMode: true, port: PORT, host: '0.0.0.0' },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    // Production static serving
    const distPath = path.join(__dirname, 'dist');
    app.use(express.static(distPath));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer();
