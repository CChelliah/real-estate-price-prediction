import React, { Component } from 'react'
import { Switch, Route, Redirect, withRouter, Link } from 'react-router-dom'
import { Breadcrumb, BreadcrumbItem, Button, Col, Row, Form, FormControl, FormGroup, Label, Input, Table} from 'reactstrap'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faEdit, faTrashAlt} from '@fortawesome/free-solid-svg-icons'



class Display extends Component {
  constructor (props) {
    super(props) 
    this.state = {
      users: []
    }
  }

  componentDidMount () {
    fetch('http://localhost:5000')
      .then(response => response.json())
      .then(data => this.setState({ users: data }))
    this.state.users = JSON.stringify(this.state.users)
  }

  render () {
    console.log(this.state.users)
    return (
      <Table responsive size='sm'> 
        <thead>
          <tr>
            <th>Given Name</th>
            <th>Surname</th>
            <th>Email</th>
            <th>Phone</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {this.state.users.map((i) => (
            <tr>
              <td>{i.givenName}</td>
              <td>{i.surname}</td>
              <td>{i.email}</td>
              <td>{i.phone}</td>
              <td>
                <FontAwesomeIcon icon={faEdit} />&nbsp;
                <FontAwesomeIcon icon={faTrashAlt} />
              </td>
            </tr>
          ))}
        </tbody>
      </Table>

    )
  }
}

export default Display
