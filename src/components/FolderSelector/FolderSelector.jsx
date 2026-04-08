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

        const firstFile = files[0];
        const filePath = firstFile.path;
        if (!filePath) {
            alert("Caminho da pasta não disponível. Certifique-se de rodar o app via Electron.");
            return;
        }

        // webkitRelativePath is relative to the selected folder, e.g.:
        //   Single subject: "subjectFolder/dti_L1.nii.gz"   → depth 1 → go up 1 level
        //   Multi subject:  "parentFolder/subject/dti_L1.nii.gz" → depth 2 → go up 2 levels
        const relDepth = (firstFile.webkitRelativePath || '').split('/').filter(Boolean).length;
        const levelsUp = relDepth >= 2 ? 2 : 1;

        const sep = filePath.includes('/') ? '/' : '\\';
        const folderPath = filePath.split(sep).slice(0, -levelsUp).join(sep);

        if (!folderPath) return;

        document.querySelector(`#check_${props.id}`).style.display = "flex";
        setFolderPath(folderPath);
        savePath(folderPath);
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