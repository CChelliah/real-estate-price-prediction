import React, { Component } from 'react'
import { Switch, Route, Redirect, withRouter, Link } from 'react-router-dom'
import { Breadcrumb, BreadcrumbItem, Button, Col, Row, Form, FormControl, FormGroup, Label, Input, Dropdown, DropdownToggle, DropdownItem, DropdownMenu } from 'reactstrap'

class Home extends Component {
  constructor (props) {
    super(props)
    this.state = {
      postalCode: '',
      bed: '',
      bath: '',
      car: '',
      propType: '',
      prediction: ''
    }
    this.handleSubmit = this.handleSubmit.bind(this)
    this.handleChange = this.handleChange.bind(this)
  }

  handleChange (event) {
    this.setState({ value: event.target.value })
  }

  handleSubmit (event) {
    // console.log(JSON.stringify(this.state))
    fetch('http://localhost:4005/', {
      method: 'POST', // or 'PUT'
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(this.state)
    })
      .then(response => response.json())
      .then(data => {
        this.setState({ prediction: data.prediction })
        console.log('Success:', data)
      })
      .catch((error) => {
        console.error('Error:', error)
      })
    console.log(JSON.stringify(this.state))
    event.preventDefault()
    event.target.reset()
  }

  render () {
    return (
      <div className='container'>
        <Form className='text-md-left' onSubmit={(e) => this.handleSubmit(e)}>
          <h5>Property Details</h5>
          <hr />
          <Row>
            <Col>
              <FormGroup>
                <Label>Postal Code:</Label>
                <Input type='text' name='postalCode' onChange={e => this.setState({ postalCode: e.target.value })} />
              </FormGroup>
            </Col>
            <Col>
              <FormGroup>
                <Label>Bed:</Label>
                <Input type='text' name='bed' onChange={e => this.setState({ bed: e.target.value })} />
              </FormGroup>
            </Col>
          </Row>
          <Row>
            <Col>
              <FormGroup>
                <Label>Bath:</Label>
                <Input type='text' name='bath' onChange={e => this.setState({ bath: e.target.value })} />
              </FormGroup>
            </Col>
            <Col>
              <FormGroup>
                <Label>Car Spaces:</Label>
                <Input type='text' name='car' onChange={e => this.setState({ car: e.target.value })} />
              </FormGroup>
            </Col>
          </Row>
          <Row>
            <Col>
              <FormGroup>
                <Label>Property Type:</Label>
                <Input type='select' name='propType' onClick={e => this.setState({ propType: e.target.value })}>
                  <option>Please select an option below...</option>
                  <option>house</option>
                  <option>duplex/semi-detached</option>
                  <option>terrace</option>
                  <option>townhouse</option>
                  <option>villa</option>
                </Input>
              </FormGroup>
            </Col>
          </Row>
          <Row>
            <Col>
              <FormGroup>
                <Button type='submit' color='success' block>Submit</Button>
              </FormGroup>
            </Col>
          </Row>
        </Form>
        <Row>
          <Col>
            <Label>Price Prediction:</Label>
            <div>{this.state.prediction}</div>
          </Col>
        </Row>

      </div>
    )
  }
}

export default Home
