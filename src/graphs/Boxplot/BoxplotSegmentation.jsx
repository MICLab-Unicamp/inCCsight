import React from 'react'
import Boxplot from './Boxplot'
import './BoxplotSegmentation.scss'

function getScalarValues(data, method, scalar) {
    return data.map(subject => parseFloat(subject[method][scalar].toFixed(6)))
}

function BoxplotSegmentation(props) {
    const ids = props.data.map(s => s["Id"])

    return (
        <div className='boxplot-container'>
            <span className='boxplot-title'>Segmentation Boxplots</span>

            <div className='boxplot-row'>
                {["FA", "MD", "RD", "AD"].map(scalar => (
                    <Boxplot
                        key={scalar}
                        title={scalar}
                        ids={ids}
                        watershed={getScalarValues(props.data, "Watershed_scalar", scalar)}
                        roqs={getScalarValues(props.data, "ROQS_scalar", scalar)}
                        cnn={getScalarValues(props.data, "santarosa_scalars", scalar)}
                        width="375"
                    />
                ))}
            </div>
        </div>
    )
}

export default BoxplotSegmentation
