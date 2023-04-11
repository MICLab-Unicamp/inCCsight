import React from 'react'
import './InfoTool.scss'

function InfoTool(props) {
  return (
    <span className='tooltip'>
        ?
        <span className='tooltip-text'>{props.text}</span>
    </span>
  )
}

export default InfoTool