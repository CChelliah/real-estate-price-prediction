var express = require('express')
var router = express.Router()
const mongoose = require('mongoose')
const bodyParser = require('body-parser')

const PriceRequest = require('../models/priceRequest')

const indexRouter = express.Router()

indexRouter.use(bodyParser.json())

indexRouter.route('/')
  .post((req, res, next) => {
    PriceRequest.create(req.body)
      .then((PriceRequest) => {
        console.log('Created user: ', PriceRequest)
        res.statusCode = 200
        res.setHeader('Content-Type', 'application/json')
        res.json(PriceRequest)
      }, (err) => next(err))
      .catch((err) => next(err))
  })

module.exports = indexRouter