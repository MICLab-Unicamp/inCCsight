import React, { useState } from 'react'
import Boxplot from './Boxplot'
import './BoxplotParcellation.scss'

function getScalarValues(subjects, method, parc_method, scalar, part) {
    const name = `${parc_method}_${scalar}_${part}`
    return subjects.map(subject => subject[method][name])
}

function BoxplotParcellation(props) {
    const [methodParcellation, setMethodParcellation] = useState("Witelson")
    const [scalarParcellation, setScalarParcellation] = useState("FA")
    const ids = props.data.map(s => s["Id"])

    return (
        <div className='boxplot-container'>
            <span className='boxplot-title'>Parcellation Boxplots</span>

            <div className='boxplot-row'>
                {["P1", "P2", "P3", "P4", "P5"].map(part => (
                    <Boxplot
                        key={part}
                        title={part}
                        ids={ids}
                        watershed={getScalarValues(props.data, "Watershed_parcellation", methodParcellation, scalarParcellation, part)}
                        roqs={getScalarValues(props.data, "ROQS_parcellation", methodParcellation, scalarParcellation, part)}
                        width="300"
                    />
                ))}
            </div>

            <div className='options-row'>
                <div className='select-group'>
                    <label>Parc. Method: </label>
                    <select onChange={e => setMethodParcellation(e.target.value)}>
                        {["Witelson", "Hofer", "Chao", "Cover", "Freesurfer"].map(m => (
                            <option key={m} value={m}>{m}</option>
                        ))}
                    </select>
                </div>

                <div className='select-group'>
                    <label>Scalar: </label>
                    <select onChange={e => setScalarParcellation(e.target.value)}>
                        {["FA", "RD", "AD", "MD"].map(s => (
                            <option key={s} value={s}>{s}</option>
                        ))}
                    </select>
                </div>
            </div>
        </div>
    )
}

export default BoxplotParcellation
