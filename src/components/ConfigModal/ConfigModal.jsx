import React, {useState} from 'react'
import './ConfigModal.scss'
import {AiOutlineClose} from 'react-icons/ai'

function closeModal(root){
    root.unmount();
}

function ConfigModal(props) {

    const [page, setPage] = useState("Graphs")
    const graphs = ["Scatter", "Line", "Bar", "Pie", "Bubble", "Table", "Sankey", "Boxplot", "Error Bar"]

    if(page === "Graphs"){
        return (
            <div id='modal-area'>
                
                <div className='modal-container'>
                    
                    <AiOutlineClose onClick={() => {closeModal(props.root)}} className="close-icon"/>
    
                    <div className='modal-body'>
    
                        <div className='modal-left'>
                            <span className='active'>Graphs and View</span>
                            <span onClick={() => {setPage("settings")}}>Settings</span>
                            <span onClick={() => {setPage("methods")}}>Methods</span>
                            <span>User Preferences</span>
                        </div>
    
                        <div className='modal-right'>
    
                            <div className='modal-right-view'>
    
                                <div className='input-group'>
    
                                    <label>Type of Graph</label>
                                    <select>
                                        {graphs.map((graph) => {
                                            return(
                                                <option value={graph}>{graph}</option>
                                            )
                                        })}
                                    </select>
    
                                </div>
    
                                <div className='input-group'>
                                    <label>Name</label>
                                    <input type="text"/>
                                </div>
    
                                <button className='btn-apply'>Apply and Save</button>
    
                            </div>
    
                        </div>
                                                
                    </div>
    
                </div>
    
            </div>
        )
    } else if(page == "methods"){
        return (
            <div id='modal-area'>
                
                <div className='modal-container'>
                    
                    <AiOutlineClose onClick={() => {closeModal(props.root)}} className="close-icon"/>
    
                    <div className='modal-body'>
    
                        <div className='modal-left'>
                            <span className='active'>Graphs and View</span>
                            <span onClick={() => {setPage("settings")}}>Settings</span>
                            <span>Methods</span>
                            <span>User Preferences</span>
                        </div>
    
                        <div className='modal-right'>
    
                            <div className='modal-right-view'>
    
                                <div className='input-group'>
    
                                    <label>Method of</label>
                                    <select>
                                        <option>Segmentation</option>
                                        <option>Parcellation</option>
                                    </select>
    
                                </div>
    
                                <div className='input-group'>
                                    <label>Select  Method</label>
                                    <input type="file"/>
                                </div>
    
                                <button className='btn-apply'>Apply and Save</button>
    
                            </div>
    
                        </div>
                                                
                    </div>
    
                </div>
    
            </div>
        )
    }

}

export default ConfigModal