import React, {useState} from 'react'
import Plot from 'react-plotly.js'
import './Scatter.scss'

function getAllPoints(data, method, scalar){
    let values = []

    for(let i = 0; i !== data.length; i++){
        let value = data[i][method][scalar]
        values.push(value)
    }

    return values
}

function Scatter(props) {

    const [scalarX, setScalarX] = useState("FA")
    const [scalarY, setScalarY] = useState("MD")

    function changeScalarX(){
        let value = document.querySelector("#scalarValueX").value
        setScalarX(value)
    }

    function changeScalarY(){
        let value = document.querySelector("#scalarValueY").value
        setScalarY(value)
    }

    let watershed = {
        x: getAllPoints(props.data, "Watershed_scalar", scalarX),
        y: getAllPoints(props.data, "Watershed_scalar", scalarY),
        mode: "markers",
        type: "scatter",
        name: "Watershed"
    }

    let roqs = {
        x: getAllPoints(props.data, "ROQS_scalar", scalarX),
        y: getAllPoints(props.data, "ROQS_scalar", scalarY),
        mode: "markers",
        type: "scatter",
        name: "ROQS"
    }

    let cnn = {
        x: getAllPoints(props.data, "santarosa_scalars", scalarX),
        y: getAllPoints(props.data, "santarosa_scalars", scalarY),
        mode: "markers",
        type: "scatter",
        name: "CNN"
    }

    let watershedHistogramX = {
        x: getAllPoints(props.data, "Watershed_scalar", scalarX),
        type: "histogram",
        name: "Watershed",
        opacity: 0.5
    }

    let ROQSHistogramX = {
        x: getAllPoints(props.data, "ROQS_scalar", scalarX),
        type: "histogram",
        name: "ROQS",
        opacity: 0.5
    }

    let cnnHistogramX = {
        x: getAllPoints(props.data, "santarosa_scalars", scalarX),
        type: "histogram",
        name: "CNN",
        opacity: 0.5
    }

    let watershedHistogramY = {
        x: getAllPoints(props.data, "Watershed_scalar", scalarY),
        type: "histogram",
        name: "Watershed",
        opacity: 0.5
    }

    let ROQSHistogramY = {
        x: getAllPoints(props.data, "ROQS_scalar", scalarY),
        type: "histogram",
        name: "ROQS",
        opacity: 0.5
    }

    let cnnHistogramY = {
        x: getAllPoints(props.data, "santarosa_scalars", scalarY),
        type: "histogram",
        name: "CNN",
        opacity: 0.5
    }

    let dataHistogramX = [watershedHistogramX, ROQSHistogramX, cnnHistogramX]
    let dataHistogramY = [watershedHistogramY, ROQSHistogramY, cnnHistogramY]
    let data = [watershed, roqs, cnn]
    
    let layoutHistogramX = {
        barmode: "overlay",
        width: 800,
        height: 280,
        margin: {t: 30, b: 0},
        title: `Scalar: ${scalarX}`
    }

    let layoutHistogramY = {
        barmode: "overlay",
        width: 800,
        height: 280,
        margin: {t: 30, b: 0},
        title: `Scalar: ${scalarY}`
    }

    let layout = {
                plot_bgcolor: '#E5ECF6',
                height: 700,
                margin: {t: 10},
                yaxis: {gridcolor: 'rgb(255, 255, 255)', title: scalarY},
                xaxis: {gridcolor: 'rgb(255, 255, 255)', title: scalarX},
                legend: { orientation: "h" }
                }

    return (
        <div className='scatter-container'>
            
            <span className='scatter-title'>Segmentation Statistics</span>

            <div className='select-row'>

                <div className='select-scalar'>
                    <span>Scalar X</span>
                    <select className='select' onChange={changeScalarX} id="scalarValueX">
                        <option value="FA">FA</option>
                        <option value="MD">MD</option>
                        <option value="RD">RD</option>
                        <option value="AD">AD</option>
                    </select>
                </div>

                <div className='select-scalar'>
                    <span>Scalar Y</span>
                    <select className='select' onChange={changeScalarY} id="scalarValueY">
                        <option value="FA">FA</option>
                        <option value="MD">MD</option>
                        <option value="RD">RD</option>
                        <option value="AD">AD</option>
                    </select>
                </div>

            </div>

            <div className='scatter-row'>
                <Plot data={data} layout={layout}/>
                
                <div className='scatter-col'>   
                    <Plot data={dataHistogramX} layout={layoutHistogramX}/>
                    <Plot data={dataHistogramY} layout={layoutHistogramY}/>
                </div>

            </div>

        </div>
    )
}

export default Scatter