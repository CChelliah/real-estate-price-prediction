import React, { Component } from 'react'
import { Collapse, Navbar, NavbarToggler, NavbarBrand, Nav, NavItem, UncontrolledDropdown, DropdownToggle, DropdownMenu, DropdownItem, NavbarText } from 'reactstrap'
import { NavLink } from 'react-router-dom'

class Header extends Component {
  render () {
    return (
      <>
        <Navbar light expand="xs">
          <div className='container header'>
            <Nav className="mr-auto" navbar>
              <NavItem>
                <NavLink className='nav-link' to='/home'>
                  <span className='fa fa-home fa-lg' />Add
                </NavLink>
              </NavItem>
              <NavItem>
                <NavLink className='nav-link' to='/display'>
                  <span className='fa fa-info fa-lg' />Display
                </NavLink>
              </NavItem>
            </Nav>
          </div>
        </Navbar>
      </>
    )
  }
}

export default Header
