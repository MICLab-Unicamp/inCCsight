import { dialog } from 'electron';
import React, { useState } from 'react';

function SelectFolderInput() {
    const [selectedFolder, setSelectedFolder] = useState('');

    const handleOpenFolderDialog = () => {
        dialog.showOpenDialog({ properties: ['openDirectory'] }).then(result => {
            if (!result.canceled) {
                setSelectedFolder(result.filePaths[0]);
            }
        });
    };

    return (
        <div>
            <button onClick={handleOpenFolderDialog}>Selecionar pasta</button>
            {selectedFolder && <p>Pasta selecionada: {selectedFolder}</p>}
        </div>
    );
}

export default SelectFolderInput