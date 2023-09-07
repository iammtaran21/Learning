//Promises

const inventory = {
  sunglasses: 1900,
  pants: 1088,
  bags: 1344
};


const myExecutor = (resolve, reject) => {
    if (inventory.sunglasses > 0) {
        resolve('Sunglasses order processed.');
    } else {
        reject('That item is sold out.');
    }
};

const orderSunglasses= () => {
  return new Promise(myExecutor);
};

const orderPromise=orderSunglasses();
console.log(orderPromise,'from hea');


//SetTimeOut

console.log("Hey");

const delayedHello = () => {
  console.log('Hello');
};

setTimeout(delayedHello, 1000);

console.log("Thea");

//library.js

const {checkInventory} = require('./library.js');

const order = [['sunglasses', 190], ['bags', 200]];

// Write your code below:
const handleSuccess = (resolvedValue) => {
  console.log(resolvedValue);
};

const handleFailure = (rejectReason) => {
  console.log(rejectReason);
};

checkInventory(order)
  .then(handleSuccess, handleFailure);
