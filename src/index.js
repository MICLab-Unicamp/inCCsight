import React from 'react';
import ReactDOM from 'react-dom/client';

import Enter from "./components/Enter/Enter";
import Home from './pages/Home'
import Loading from './components/Loading/Loading'
import './styles/global.scss'

import {
  createBrowserRouter,
  RouterProvider,
} from "react-router-dom";

const root = ReactDOM.createRoot(document.getElementById('root'));

const router = createBrowserRouter([
  {
    path: "/",
    element: <div className='container'> <Enter /></div>
  },
  {
    path: "/Loading",
    element: <div className='container'><Home /></div>
  },
  {
    path: "/Home",
    element: <div className='container'><Home /></div>
  }
])

root.render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>
);