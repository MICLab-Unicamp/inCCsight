import React, {useState} from 'react'
import Boxplot from './Boxplot'
import './BoxplotParcellation.scss'

function getScalarValues(subjects, method, parc_method, scalar, part){

    let name = `${parc_method}_${scalar}_${part}`
    let values = subjects.map((subject) => {
                    return subject[method][name]
                })
    return values
}

function BoxplotParcellation(props) {

    let [methodParcellation, setMethodParcellation] = useState("Witelson")
    let [scalarParcellation, setScalarParcellation] = useState("FA")

    function changeMethod(){
        let value = document.querySelector("#methodParcellationValue").value
        setMethodParcellation(value)
    }

    function changeScalarParcellation(){
        let value = document.querySelector("#parcellationSelect").value
        console.log(value)
        setScalarParcellation(value)
    }
    
    return (

        <div className='boxplot-container'>
            <span className='boxplot-title'>Parcellation Boxplots</span>

            <div className='boxplot-row'>

                <Boxplot title="P1" watershed={getScalarValues(props.data, "Watershed_parcellation", methodParcellation, scalarParcellation, "P1")} roqs={getScalarValues(props.data, "ROQS_parcellation", methodParcellation, scalarParcellation, "P1")} width="300"/>
                <Boxplot title="P2" watershed={getScalarValues(props.data, "Watershed_parcellation", methodParcellation, scalarParcellation, "P2")} roqs={getScalarValues(props.data, "ROQS_parcellation", methodParcellation, scalarParcellation, "P2")} width="300"/>
                <Boxplot title="P3" watershed={getScalarValues(props.data, "Watershed_parcellation", methodParcellation, scalarParcellation, "P3")} roqs={getScalarValues(props.data, "ROQS_parcellation", methodParcellation, scalarParcellation, "P3")} width="300"/>
                <Boxplot title="P4" watershed={getScalarValues(props.data, "Watershed_parcellation", methodParcellation, scalarParcellation, "P4")} roqs={getScalarValues(props.data, "ROQS_parcellation", methodParcellation, scalarParcellation, "P4")} width="300"/>
                <Boxplot title="P5" watershed={getScalarValues(props.data, "Watershed_parcellation", methodParcellation, scalarParcellation, "P5")} roqs={getScalarValues(props.data, "ROQS_parcellation", methodParcellation, scalarParcellation, "P5")} width="300"/>

            </div>

            <div className='options-row'>  
        
                <div className='select-group'>
                    <label>Parc. Method: </label>
                    <select id="methodParcellationValue" onChange={changeMethod}>
                        <option value="Witelson">Witelson</option>
                        <option value="Hofer">Hofer</option>
                        <option value="Chao">Chao</option>
                        <option value="Cover">Cover</option>
                        <option value="Freesurfer">Freesurfer</option>
                    </select>
                </div>

                <div className='select-group'>
                    <label>Scalar: </label>
                    <select id="parcellationSelect" onChange={changeScalarParcellation}>
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

export default BoxplotParcellation