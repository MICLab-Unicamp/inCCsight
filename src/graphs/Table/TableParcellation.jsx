import React, {useState} from 'react'
import Plot from 'react-plotly.js'
import './TableParcellation.scss'

import InfoTool from '../../components/InfoTool/InfoTool'

function getMeanValues(subjects, method, parc_method, scalar, part){
    let name = `${parc_method}_${scalar}_${part}`
    let value = 0
    subjects.map((subject) => {
        return(
            value += subject[method][name]
        )
    })

    value /= (subjects.length);
    return value.toFixed(6)
}

function TableParcellation(props) {

    const [methodParcellation, setMethodParcellation] = useState("Witelson")
    const [scalar, setScalar] = useState("FA")

    function changeMethod(){
        let value = document.querySelector("#methodValue").value
        setMethodParcellation(value)
    }

    function changeScalar(){
        let value = document.querySelector("#scalarValueParcellation").value
        setScalar(value)
    }

    let headers = ["Method", "P1", "P2", "P3", "P4", "P5"]
    let subjects = props.data

    let cols = [["ROQS", "Watershed-Based"]]

    for(let i = 1; i !== headers.length; i++){
        let v1 = getMeanValues(subjects, "ROQS_parcellation", methodParcellation, scalar, headers[i])
        let v2 = getMeanValues(subjects, "Watershed_parcellation", methodParcellation, scalar, headers[i])
        cols.push([v1, v2])
    }
    

    let data = [{
        type: "table",
        header: {
            values: headers,
            align: ["center"],
            line: {width: 1, color: 'black'},
            fill: {color: "grey"},
            font: {family: "Arial", size: 14, color: "white"}
        },
        cells: {
            values: cols,
            height: 30,
            align: ["center", "center"],
            line: {width: 1, color: 'black'},
            font: {family: "Arial", size: 12, color: "black"}    
        }
    }]

    let layout = {width: "50%", height: 130, margin: {t: 10, b: 0, l: 10, r: 10}, 
    paper_bgcolor: props.bg_color}


    if(props.type === "2D"){

        return (
            <div className='table-field'>
                    
                <div className='table-row'>
                    <span className={`table-title ${props.color}`}>Parcellation Data <InfoTool text="Comparison of the average of the values ​​obtained from each part by the installment in each method."/></span>
                    <button className='btn-export'>Export</button>
                </div>
    
                <Plot data={data} layout={layout}/>
    
                <div className='options-row'>  
            
                    <div className='select-group'>
                        <label className={props.color}>Parc. Method: </label>
                        <select id="methodValue" onChange={changeMethod}>
                            <option value="Witelson">Witelson</option>
                            <option value="Hofer">Hofer</option>
                            <option value="Chao">Chao</option>
                            <option value="Cover">Cover</option>
                            <option value="Freesurfer">Freesurfer</option>
                        </select>
                    </div>
    
                    <div className='select-group'>
                        <label className={props.color}>Scalar: </label>
                        <select id="scalarValueParcellation" onChange={changeScalar}>
                            <option value="FA">FA</option>
                            <option value="RD">RD</option>
                            <option value="AD">AD</option>
                            <option value="MD">MD</option>
                        </select>
                    </div>
            
                </div>
    
            </div>
        )
    } else if(props.type === "3D"){

        let cols = [["CNN-Based"]]
        let data = [{
            type: "table",
            header: {
                values: headers,
                align: ["center"],
                line: {width: 1, color: 'black'},
                fill: {color: "grey"},
                font: {family: "Arial", size: 14, color: "white"}
            },
            cells: {
                values: cols,
                height: 30,
                align: ["center", "center"],
                line: {width: 1, color: 'black'},
                font: {family: "Arial", size: 12, color: "black"}    
            }
        }]

        return (
            <div className='table-field'>
                    
                <div className='table-row'>
                    <span className={`table-title ${props.color}`}>Parcellation Data <InfoTool text="Comparison of the average of the values ​​obtained from each part by the installment in each method."/></span>
                    <button className='btn-export'>Export</button>
                </div>
    
                <Plot data={data} layout={layout}/>
    
                <div className='options-row'>  
            
                    <div className='select-group'>
                        <label className={props.color}>Parc. Method: </label>
                        <select id="methodValue" onChange={changeMethod}>
                            <option value="Witelson">Witelson</option>
                            <option value="Hofer">Hofer</option>
                            <option value="Chao">Chao</option>
                            <option value="Cover">Cover</option>
                            <option value="Freesurfer">Freesurfer</option>
                        </select>
                    </div>
    
                    <div className='select-group'>
                        <label className={props.color}>Scalar: </label>
                        <select id="scalarValueParcellation" onChange={changeScalar}>
                            <option value="FA">FA</option>
                            <option value="RD">RD</option>
                            <option value="AD">AD</option>
                            <option value="MD">MD</option>
                        </select>
                    </div>
            
                </div>
    
            </div>
        )
    }

}

export default TableParcellation