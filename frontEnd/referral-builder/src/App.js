import React, { Component } from 'react'
import { Navbar, NavbarBrand } from 'reactstrap'
import './App.css'
import { BrowserRouter } from 'react-router-dom'
import { Provider } from 'react-redux'
import Main from './components/MainComponent'

class App extends Component {
  render () {
    return (
      <BrowserRouter>
        <div className='App'>
          <Main />
        </div>
      </BrowserRouter>
    )
  }
}

export default App
