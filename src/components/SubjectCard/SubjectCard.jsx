import React from 'react'
import './SubjectCard.scss'

function qcDotClass(qc) {
    if (!qc || qc.ROQS == null) return 'qc-dot qc-na'
    if (qc.ROQS.flag === true) return 'qc-dot qc-fail'
    if (qc.ROQS.flag === false) return 'qc-dot qc-pass'
    return 'qc-dot qc-na'
}

function SubjectCard(props) {
    return (
        <div className='subject-card' id={props.id} onClick={() => {props.onClick(props.name)}}>
            {props.name}
            {props.qc !== undefined && <span className={qcDotClass(props.qc)} title={
                props.qc?.ROQS?.prob != null
                    ? `QC ROQS: ${props.qc.ROQS.flag ? 'FAIL' : 'PASS'} (${(props.qc.ROQS.prob * 100).toFixed(1)}%)`
                    : 'QC N/A'
            } />}
        </div>
    )
}

export default SubjectCard