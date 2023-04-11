import React, {useState} from 'react'
import Plot from 'react-plotly.js'
import InfoTool from '../../components/InfoTool/InfoTool'
import './TableSegmentation.scss'

import { TbEyeFilled, TbEyeOff } from 'react-icons/tb'

function getMeanValues(subjects, method, scalar){
    let value = 0
    subjects.map((subject) => {
        return(
            value += subject[method][scalar]
        )
    })

    value /= (subjects.length);
    return value.toFixed(6)
}

function TableSegmentation(props) {

    let [view, setView] = useState("hide")
    let [size, setSize] = useState(130)
    let [iconEye, setIconEye] = useState(<TbEyeFilled />)

    function changeShowSegmentation(type) {
        let value = document.querySelector("#showView2D").value
        console.log(value);
    }

    function teste(){
        if(view == "hide"){
            setView("show")
            setIconEye(<TbEyeOff />)
        } else{
            setView("hide")
            setIconEye(<TbEyeFilled />)
        }
    }

        //if(view === "show"){
        //    setSize(160)
        //} else{
        //    setSize(130)
        //}

    /* Remover depois */
    let joany = [0.6556650233259003, 0.248837988995943, 0.0010308019022138776, 0.0004797977367285885, 0.0006192081277246568, 0.0005655540233389842, 0.0018539894511923193, 0.0004405055082694843]


    let headers = []
    /* Selecionando os dados */
    let subjects = props.data
    if(view === "show"){
        headers = ["Method", "FA", "FA StdDev","MD", "MD StdDev", "RD", "RD StdDev", "AD", "AD StdDev"]
    } else if(view === "hide"){
        headers = ["Method", "FA", "MD", "RD", "AD"]
    }

    let cols = [["ROQS", "Watershed-Based", "CNN-Based"]]

    for(let i = 1; i !== headers.length; i++){
        let v1 = getMeanValues(subjects, "ROQS_scalar", headers[i])
        let v2 = getMeanValues(subjects, "Watershed_scalar", headers[i])
        let v3 = getMeanValues(subjects, "santarosa_scalars", headers[i])
        cols.push([v1, v2, v3])
    }
    
    /* Ajustes da tabela */
    
    let layout = {width: "50%", height: size, margin: {t: 10, b: 0, l: 10, r: 10}, paper_bgcolor: props.bg_color}

    if(props.type === "2D"){

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

        return(

            <div className='table-field'>
                    
                <div className='table-row'>
                    <span className={`table-title ${props.color}`}>Segmentation Data <InfoTool text="Comparison of the mean values ​​obtained by segmentation in each method."/></span>
                    <button className='btn-export'>Export</button>
                </div>
    
                <Plot data={data} layout={layout}/>
            
                <div className='options-row'>  
            
                    <div className='select-group'>
                        <label className={props.color}>Std. Dev: </label>

                        <button onClick={teste} className="btn-icon">{iconEye}</button>

                    </div>
            
                </div>
    
            </div>
        )
    } else if(props.type === "3D"){

        let joany = [0.6556650233259003, 0.248837988995943, 0.0010308019022138776, 0.0004797977367285885, 0.0006192081277246568, 0.0005655540233389842, 0.0018539894511923193, 0.0004405055082694843]
        let cols = [["CNN-Based"]]

        for(let i = 1; i !== headers.length; i++){
            let v1 = getMeanValues(subjects, "santarosa_scalars", headers[i])
            cols.push([v1])
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

        return(

            <div className='table-field'>
                    
                <div className='table-row'>
                    <span className={`table-title ${props.color}`}>Segmentation Data <InfoTool text="Comparison of the mean values ​​obtained by segmentation in each method."/></span>
                    <button className='btn-export'>Export</button>
                </div>
    
                <Plot data={data} layout={layout}/>
            
                <div className='options-row'>  
            
                    <div className='select-group'>
                        <label className={props.color}>Std. Dev: </label>
                        <select onChange={() => {changeShowSegmentation("3D")}} id="show">
                            <option value="hide">Hide</option>
                            <option value="show">Show</option>
                        </select>
                    </div>
            
                </div>
    
            </div>
        )
    }

}

export default TableSegmentation