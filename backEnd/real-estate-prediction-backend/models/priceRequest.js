const mongoose = require('mongoose')
const Schema = mongoose.Schema

const priceRequestSchema = new Schema({
    postalCode: {
        type: String
    },
    bed: {
        type: String
    },
    bath: {
        type: String
    },
    car: {
        type: String
    },
    propType: {
        type: String
    },
}, {
    timestamps: true
})

var PriceRequest = mongoose.model('PriceRequest', priceRequestSchema)

module.exports = PriceRequest