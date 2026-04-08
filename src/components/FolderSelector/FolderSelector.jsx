import React, { useRef } from "react";
import './FolderSelector.scss'
import { TbFolder, TbX, TbChecks } from 'react-icons/tb'

function FolderSelector({ id, path, groupName, onUpdate, onRemove, colorIndex }) {

    const inputRef = useRef(null);

    function handleNameChange(e) {
        onUpdate({ groupName: e.target.value });
    }

    function handleFolderChange(event) {
        const files = event.target.files;
        if (!files || files.length === 0) return;

        const firstFile = files[0];
        const filePath = firstFile.path;
        if (!filePath) {
            alert("Caminho da pasta não disponível. Certifique-se de rodar o app via Electron.");
            return;
        }

        // Determine how many levels to go up to get the selected folder
        const relDepth = (firstFile.webkitRelativePath || '').split('/').filter(Boolean).length;
        const levelsUp = relDepth >= 2 ? 2 : 1;
        const sep = filePath.includes('/') ? '/' : '\\';
        const resolved = filePath.split(sep).slice(0, -levelsUp).join(sep);
        if (!resolved) return;

        onUpdate({ path: resolved });
    }

    const folderLabel = path
        ? path.split(/[\\/]/).pop() || path
        : 'Selecionar pasta...';

    const COLOR_CLASSES = ['color-0', 'color-1', 'color-2', 'color-3', 'color-4', 'color-5'];
    const colorClass = COLOR_CLASSES[colorIndex % COLOR_CLASSES.length];

    return (
        <div className={`folder-row ${colorClass}`}>
            <div className={`group-stripe`} />

            <input
                className="group-name-input"
                placeholder="Nome do grupo (ex: Controle)"
                value={groupName}
                onChange={handleNameChange}
            />

            <button
                type="button"
                className={`folder-picker-btn ${path ? 'has-path' : ''}`}
                onClick={() => inputRef.current?.click()}
                title={path || 'Nenhuma pasta selecionada'}
            >
                <input
                    type="file"
                    webkitdirectory="true"
                    onChange={handleFolderChange}
                    ref={inputRef}
                    style={{ display: "none" }}
                />
                <TbFolder className="folder-icon" />
                <span className="folder-label">{folderLabel}</span>
                {path && <TbChecks className="check-icon" />}
            </button>

            {onRemove && (
                <button className="remove-btn" onClick={onRemove} title="Remover grupo">
                    <TbX />
                </button>
            )}
        </div>
    );
}

export default FolderSelector;
