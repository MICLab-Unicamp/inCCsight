import React, {useState} from 'react'
import './Question.scss'

function Question(props) {
    const [expanded, setExpanded] = useState(false);
  
    const toggleExpanded = () => {
      setExpanded(!expanded);
    };
  
    return (
      <div onClick={toggleExpanded} className="question-container">
        <h3>{props.question}</h3>
        {expanded && <p>{props.response}</p>}
      </div>
    );
  }
export default Question