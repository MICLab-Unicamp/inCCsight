import React, {useState} from 'react'
import Plot from 'react-plotly.js'
import './Midline.scss'

function getMeanPointsValue(data, method, scalar){

    let size = data[0]["Watershed_midlines"]["FA"].length
    let values = []

    for(let p = 0; p !== size; p++){
        let p_sum = 0
        for(let i = 0; i !== data.length; i++){
            p_sum += data[i][method][scalar][p]
        }    
        let p_value = p_sum / size
        values.push(p_value)
    }

    return values
}

function getMeanThicknessValue(data, method){
    let size = data[0]["Watershed_thickness"].length
    let values = []

    for(let p = 0; p !== size; p++){
        let p_sum = 0
        for(let i = 0; i !== data.length; i++){
            p_sum += data[i][method][p]
        }    
        let p_value = p_sum / size
        values.push(p_value)
    }
    return values
}

function Midline(props) {

    const [scalar, setScalar] = useState("FA")

    function changeScalar(){
        let value = document.querySelector("#scalarValue").value
        setScalar(value)
    }

    let roqs = {}
    let watershed = {}

    if(scalar !== "Thickness"){
        roqs = {
            y: getMeanPointsValue(props.data, "ROQS_midlines", scalar),
            modes: "lines",
            name: "ROQS"
        }
    
        watershed = {
            y: getMeanPointsValue(props.data, "Watershed_midlines", scalar),
            modes: "lines",
            name: "Watershed"
        }
    } else{
        roqs = {
            y: getMeanThicknessValue(props.data, "Watershed_thickness"),
            modes: "lines",
            name: "ROQS"
        }
    
        watershed = {
            y: getMeanThicknessValue(props.data, "ROQS_thickness"),
            modes: "lines",
            name: "Watershed"
        }
    }

    let data = [roqs, watershed]

    let layout = {  
                    title: "Midline Plots", 
                    height: 420, width: 660, margin: {t: 40, l: 50, r: 10}, 
                    legend: { orientation: "h" , x: 1, y: 1.1, xanchor: 'right'},
                    plot_bgcolor: '#E5ECF6',
                    yaxis: {
                        gridcolor: 'rgb(255, 255, 255)'
                    },
                    xaxis: {
                        gridcolor: 'rgb(255, 255, 255)',
                        title: "Points Along CC Body"
                    }
                }

    return (
        <div className='midline-container'>
            
            <Plot data={data} layout={layout}/>

            <div className='select-scalar'>
                <span>Scalar</span>
                <select className='select' onChange={changeScalar} id="scalarValue">
                    <option value="FA">FA</option>
                    <option value="MD">MD</option>
                    <option value="RD">RD</option>
                    <option value="AD">AD</option>
                    <option value="Thickness">Thickness</option>
                </select>
            </div>

        </div>
    )
}

export default Midline