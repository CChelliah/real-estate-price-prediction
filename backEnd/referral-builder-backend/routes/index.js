var express = require('express')
var router = express.Router()
const mongoose = require('mongoose')
const bodyParser = require('body-parser')

const Users = require('../models/users')

const indexRouter = express.Router()

indexRouter.use(bodyParser.json())

indexRouter.route('/')
  .get((req, res, next) => {
    console.log("GET ROUTE")
    Users.find({})
      .then((users) => {
        console.log("Found users", users)
        res.statusCode = 200
        res.setHeader('Content-Type', 'application/json')
        res.json(users)
      }, (err) => next(err))
      .catch((err) => next(err))
  })

  .post((req, res, next) => {
    Users.create(req.body)
      .then((users) => {
        console.log('Created user: ', users)
        res.statusCode = 200
        res.setHeader('Content-Type', 'application/json')
        res.json(user)
      }, (err) => next(err))
      .catch((err) => next(err))
  })

module.exports = indexRouter
