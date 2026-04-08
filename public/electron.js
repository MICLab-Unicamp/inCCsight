const { app, BrowserWindow, ipcMain, Menu } = require('electron');
const remote = require('@electron/remote/main');
const path = require('path');
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