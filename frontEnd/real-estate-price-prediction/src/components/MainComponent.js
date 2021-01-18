import React, { Component } from 'react'
import { Switch, Route, Redirect, withRouter } from 'react-router-dom'
import { connect } from 'react-redux'
import { actions } from 'react-redux-form'
import { TransitionGroup, CSSTransition } from 'react-transition-group'
import Header from './HeaderComponent'
import Home from './HomeComponent'
import Display from './DisplayComponent'
import '../App.css'

const users = 'test'

class Main extends Component {
  constructor (props) {
    super(props)
  }

  render () {
    return (
      <div className='container'>
        <Header />
        <Route path='/home' component={Home} />
        <Route path='/display' component={Display} />
        <Redirect to='/home' />
      </div>
    )
  }
}

export default Main
