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
        
        const selectedFolder = event.target.files[0].path;

        let lastSlashIndex = selectedFolder.lastIndexOf("/");
        let penultimateSlashIndex = selectedFolder.lastIndexOf("/", lastSlashIndex - 1);
        let folderPath = selectedFolder.substring(0, penultimateSlashIndex);
        
        if(folderPath.length == 0){
            lastSlashIndex = selectedFolder.lastIndexOf("\\");
            penultimateSlashIndex = selectedFolder.lastIndexOf("\\", lastSlashIndex - 1);
            folderPath = selectedFolder.substring(0, penultimateSlashIndex);
        }

        let check = document.querySelector(`#check_${props.id}`)
        check.style.display = "flex"

        if (selectedFolder) {
            setFolderPath(folderPath);
            savePath(folderPath)
        }
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