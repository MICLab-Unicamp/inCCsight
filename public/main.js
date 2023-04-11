const { app, BrowserWindow, ipcMain, Menu } = require('electron');

let win; // Armazena a referência da janela "win"

function createWindow() {
  win = new BrowserWindow({
    width: 800,
    height: 600,
    frame: true,
    backgroundColor: '#80000000',
    resizable: false,
    webPreferences: {
      nodeIntegration: true,
      enableRemoteModule: true,
      contextIsolation: false,
    },

  });
  // Menu.setApplicationMenu(null)

  win.loadURL('http://localhost:3000');
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
      nodeIntegration: true
    },
    show: false
  });
  newWindow.maximize()
  newWindow.show()

  newWindow.loadURL('http://localhost:3000/Home');

  newWindow.on('closed', () => {
    newWindow = null;
  });
  
  win.close(); // Fecha a janela "win"
});
