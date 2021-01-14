import React, { Component } from 'react'
import { Switch, Route, Redirect, withRouter, Link } from 'react-router-dom'
import { Breadcrumb, BreadcrumbItem, Button, Col, Row, Form, FormControl, FormGroup, Label, Input } from 'reactstrap'

class Home extends Component {
  constructor (props) {
    super(props)
    this.state = {
      givenName: '',
      surname: '',
      email: '',
      phone: '',
      homeName: '',
      street: '',
      suburb: '',
      state: '',
      postcode: '',
      country: ''
    }
    this.handleSubmit = this.handleSubmit.bind(this)
    this.handleChange = this.handleChange.bind(this)
  }

  handleChange (event) {
    this.setState({ value: event.target.value })
  }

  handleSubmit (event) {
    console.log(JSON.stringify(this.state))

    fetch('http://0.0.0.0:5000/', {
      method: 'POST', // or 'PUT'
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(this.state)
    })
      .then(response => response.json())
      .then(data => {
        console.log('Success:', data)
      })
      .catch((error) => {
        console.error('Error:', error)
      })
    event.preventDefault()
    event.target.reset()
  }

  render () {
    return (
      <div className='container'>
        <Form className='text-md-left' onSubmit={(e) => this.handleSubmit(e)}>
          <h5>Personal Details</h5>
          <hr />
          <Row>
            <Col>
              <FormGroup>
                <Label>Given Name:</Label>
                <Input type='text' name='givenName' onChange={e => this.setState({ givenName: e.target.value })} />
              </FormGroup>
            </Col>
            <Col>
              <FormGroup>
                <Label>Surname:</Label>
                <Input type='text' name='surname' onChange={e => this.setState({ surname: e.target.value })} />
              </FormGroup>
            </Col>
          </Row>
          <Row>
            <Col>
              <FormGroup>
                <Label>Email:</Label>
                <Input type='text' name='email' onChange={e => this.setState({ email: e.target.value })} />
              </FormGroup>
            </Col>
            <Col>
              <FormGroup>
                <Label>Phone:</Label>
                <Input type='text' name='phone' onChange={e => this.setState({ phone: e.target.value })} />
              </FormGroup>
            </Col>
          </Row>
          <h5>Address</h5>
          <hr />
          <Row>
            <Col>
              <FormGroup>
                <Label>Home Name OR #:</Label>
                <Input type='text' name='homeName' onChange={e => this.setState({ homeName: e.target.value })} />
              </FormGroup>
            </Col>
            <Col>
              <FormGroup>
                <Label>Street:</Label>
                <Input type='text' name='street' onChange={e => this.setState({ street: e.target.value })} />
              </FormGroup>
            </Col>
          </Row>
          <Row>
            <Col>
              <FormGroup>
                <Label>Suburb:</Label>
                <Input type='text' name='suburb' onChange={e => this.setState({ suburb: e.target.value })} />
              </FormGroup>
            </Col>
            <Col>
              <FormGroup>
                <Label>State</Label>
                <Input type='text' name='state' onChange={e => this.setState({ state: e.target.value })} />
              </FormGroup>
            </Col>
          </Row>
          <Row>
            <Col>
              <FormGroup>
                <Label>Postcode:</Label>
                <Input type='text' name='postcode' onChange={e => this.setState({ postcode: e.target.value })} />
              </FormGroup>
            </Col>
            <Col>
              <FormGroup>
                <Label>Country:</Label>
                <Input type='text' name='country' onChange={e => this.setState({ country: e.target.value })} />
              </FormGroup>
            </Col>
          </Row>
          <Row>
            <Col>
              <FormGroup>
                <Button color='outline-secondary' block>Upload Avatar</Button>
              </FormGroup>
            </Col>
            <Col>
              <FormGroup>
                <Button type='submit' color='success' block>Create Referral</Button>
              </FormGroup>
            </Col>
          </Row>
        </Form>
      </div>
    )
  }
}

export default Home
