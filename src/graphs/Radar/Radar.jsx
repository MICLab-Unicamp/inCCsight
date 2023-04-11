import React, {useState} from 'react'
import Plot from 'react-plotly.js'
import './Radar.scss'

function getMeanValues(subjects, method, parc_method, scalar, part){
    let name = `${parc_method}_${scalar}_${part}`
    let value = 0
    subjects.map((subject) => {
        return(
            value += subject[method][name]
        )
    })

    value /= (subjects.length);
    return parseFloat(value.toFixed(6))
}

function getAllValues(subjects, method, parc_method, scalar){
    let values = []
    for(let i = 1; i !== 6; i++){
        values.push(getMeanValues(subjects, method, parc_method, scalar, `P${i}`))
    }
    return values
}

function Radar(props) {
    const [methodRadar, setMethodRadar] = useState("Witelson")
    const [scalarRadar, setScalarRadar] = useState("FA")

    function changeMethodRadar(){
        let value = document.querySelector("#methodRadar").value
        setMethodRadar(value)
    }

    function changeScalarRadar(){
        let value = document.querySelector("#scalarRadar").value
        setScalarRadar(value)
    }

    let watershed = {
        type: 'scatterpolar',
        r: getAllValues(props.data, "Watershed_parcellation", methodRadar, scalarRadar),
        theta: ['P1', 'P2','P3', 'P4', 'P5'],
        fill: 'toself',
        name: "Watershed"
    }

    let roqs = {
        type: 'scatterpolar',
        r: getAllValues(props.data, "ROQS_parcellation", methodRadar, scalarRadar),
        theta: ['P1', 'P2','P3', 'P4', 'P5'],
        fill: 'toself',
        name: "ROQS"
    }

    let data = [watershed, roqs]

    let layout = {
        title: "Radar Parcellation",
        legend: {orientation: "h"},
        
    }

    return (
        <div className='radar-container'>
            <Plot data={data} layout={layout}/>
            
            <div className='options-col'>  
        
                <div className='select-group'>
                    <label>Parc. Method: </label>
                    <select id="methodRadar" onChange={changeMethodRadar}>
                        <option value="Witelson">Witelson</option>
                        <option value="Hofer">Hofer</option>
                        <option value="Chao">Chao</option>
                        <option value="Cover">Cover</option>
                        <option value="Freesurfer">Freesurfer</option>
                    </select>
                </div>

                <div className='select-group'>
                    <label>Scalar: </label>
                    <select id="scalarRadar" onChange={changeScalarRadar}>
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

export default Radar