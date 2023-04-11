import React from 'react'
import './SubjectCard.scss'

function SubjectCard(props) {
    return (
        <div className='subject-card' id={props.id} onClick={() => {props.onClick(props.name)}}>
            {props.name}
        </div>
    )
}

export default SubjectCard