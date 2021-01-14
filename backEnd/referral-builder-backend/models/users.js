const mongoose = require('mongoose')
const Schema = mongoose.Schema

const userSchema = new Schema({
  givenName: {
    type: String
  },
  surname: {
    type: String
  },
  email: {
    type: String
  },
  phone: {
    type: String
  },
  homeName: {
    type: String
  },
  street: {
    type: String
  },
  suburb: {
    type: String
  },
  state: {
    type: String
  },
  postcode: {
    type: String
  },
  country: {
    type: String
  }
}, {
  timestamps: true
})

var Users = mongoose.model('User', userSchema)

module.exports = Users
