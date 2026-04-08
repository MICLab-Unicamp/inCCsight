const { app, BrowserWindow, ipcMain, Menu } = require('electron');
const remote = require('@electron/remote/main');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const isDev = require('electron-is-dev');

remote.initialize();

let win;

function createWindow() {
  win = new BrowserWindow({
    width: 800,
    height: 600,
    frame: true,
    backgroundColor: '#80000000',
    resizable: false,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      enableRemoteModule: false,
    },
  });

  remote.enable(win.webContents);

  const startURL = isDev
    ? 'http://localhost:3000'
    : `file://${path.join(__dirname, '../build/index.html')}`;

  win.loadURL(startURL);
}

app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// ── Pipeline IPC ──────────────────────────────────────────────────────────────

function spawnPipeline(event, args, cwd) {
  const python = process.platform === 'win32' ? 'python' : 'python3';
  const proc = spawn(python, args, { cwd, env: process.env });

  proc.stdout.on('data', d => event.reply('pipeline-output', d.toString()));
  proc.stderr.on('data', d => event.reply('pipeline-output', d.toString()));
  proc.on('close', code => event.reply('pipeline-done', { code }));
  proc.on('error', err => {
    event.reply('pipeline-output', `[ERRO] ${err.message}\n`);
    event.reply('pipeline-done', { code: 1 });
  });
}

ipcMain.on('run-pipeline', (event, { paths, groupsMap, skipCnn }) => {
  const methodsDir = path.join(__dirname, '..', 'methods');
  const groupsFile = path.join(methodsDir, 'csvs', 'groups.json');

  try {
    fs.writeFileSync(groupsFile, JSON.stringify(groupsMap, null, 2), 'utf-8');
  } catch (e) {
    event.reply('pipeline-output', `[AVISO] Não foi possível salvar groups.json: ${e.message}\n`);
  }

  const args = ['run.py', '-p', ...paths];
  if (skipCnn) args.push('--skip-cnn');
  spawnPipeline(event, args, methodsDir);
});

ipcMain.on('load-last', event => {
  const csvDir = path.join(__dirname, '..', 'methods', 'csvs');
  spawnPipeline(event, ['transformInJson.py'], csvDir);
});

// ── Open main window ──────────────────────────────────────────────────────────

ipcMain.on('open-window', () => {
  let newWindow = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    },
    show: false,
  });

  remote.enable(newWindow.webContents);

  newWindow.maximize();
  newWindow.show();

  const homeURL = isDev
    ? 'http://localhost:3000/Home'
    : `file://${path.join(__dirname, '../build/index.html#/Home')}`;

  newWindow.loadURL(homeURL);

  newWindow.on('closed', () => {
    newWindow = null;
  });

  win.close();
});