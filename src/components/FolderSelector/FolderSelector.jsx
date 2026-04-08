import React, { useState, useRef } from "react";
import './FolderSelector.scss'
import { TbFolder, TbChecks } from 'react-icons/tb'

function FolderSelector(props) {
    const [folderPath, setFolderPath] = useState([]);
    
    const inputRef = useRef(null);

    function savePath(path){
        let listPaths = JSON.parse(localStorage.getItem("folders"))
        listPaths.push(path)
        let newList = JSON.stringify(listPaths)

        localStorage.setItem("folders", newList)
    }

    function handleFolderChange(event) {
        const files = event.target.files;
        if (!files || files.length === 0) return;

        const filePath = files[0].path;
        if (!filePath) {
            alert("Caminho da pasta não disponível. Certifique-se de rodar o app via Electron.");
            return;
        }

        // Determine separator (Windows vs Unix)
        const sep = filePath.includes('/') ? '/' : '\\';
        const parts = filePath.split(sep);

        // files[0].path = /parent/selectedFolder/subject/file
        // We want /parent/selectedFolder, so drop the last two segments (file + subject)
        const folderPath = parts.slice(0, -2).join(sep);

        if (!folderPath) return;

        let check = document.querySelector(`#check_${props.id}`)
        check.style.display = "flex"

        setFolderPath(folderPath);
        savePath(folderPath)
    }

    function handleFolderButtonClick() {
        inputRef.current.click();
    }

    return (
        <div className="folder-container">
            <input className="input-text" placeholder="Ex: Control Group" id={`folder-name-${props.id}`} required/>

            <label className="input-icon" htmlFor="folder-selector">
                <input
                    type="file"
                    id={`folder-selector-${props.id}`}
                    webkitdirectory="true"
                    onChange={handleFolderChange}
                    ref={inputRef}
                    style={{ display: "none" }}
                />

                <button type="button" className="icon-button" onClick={handleFolderButtonClick}>
                    <TbFolder className="icon"/>
                    <span className="icon-text">Click for select a folder</span>
                </button>
            </label>

            <TbChecks id={`check_${props.id}`} className="check-icon"/>
        </div>
    );
}

export default FolderSelector;