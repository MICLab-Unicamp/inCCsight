import React, { useState, useEffect } from 'react'
import './Loading.scss'
import { Dna } from  'react-loader-spinner'

function Loading() {

  const phrases = [
    'The corpus callosum is the largest nerve fiber structure in the human brain. It contains about 200 million nerve fibers and weighs about 250 grams.',
    'The corpus callosum was discovered by the Italian anatomist Luigi Rolando in 1809. He called it the "cortical bridge" because it connected the cortical areas of both cerebral hemispheres.',
    "The corpus callosum is responsible for allowing the transfer of sensory, motor and cognitive information between the cerebral hemispheres. That means it's important for functions like perception, language, learning, and memory.",
    'Some medical conditions, such as epilepsy, can be treated with surgery that cuts the corpus callosum. This is known as a callosotomy and is done to prevent epileptic activity from spreading from one hemisphere to the other.',
    'Some research suggests that the size and shape of the corpus callosum may be related to gender differences. For example, studies indicate that the corpus callosum is proportionally larger in women than in men. Additionally, some research suggests that the corpus callosum may be more asymmetrical in men than in women.'
  ]
  
  const [currentPhraseIndex, setCurrentPhraseIndex] = useState(0)

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentPhraseIndex(currentIndex => (currentIndex + 1) % phrases.length)
    }, 1000)
    return () => clearInterval(interval)
  }, [phrases])

  return (
    <div className='loader-container' id='loading-container'>
        <Dna
          visible={true}
          height="80"
          width="80"
          ariaLabel="dna-loading"
          wrapperStyle={{}}
          wrapperClass="dna-wrapper"
        />
      <span className='wait-phase'>Please wait a moment while we prepare everything.</span>
      <span>Did you know?</span> 
      <span className='phrases'>{phrases[currentPhraseIndex]}</span>
    </div>
  )
}

export default Loading