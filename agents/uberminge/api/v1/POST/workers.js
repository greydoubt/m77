import axios from 'axios';

class API {
  static current_queue = [];
  static pendingPromise = false;
  static http = axios;
  static retryTimeout = null;

  static queue(promise) {
    return new Promise((resolve, reject) => {
      API.current_queue.push({
        promise,
        resolve,
        reject,
      });
      API.dequeue();
    });
  }

  static retry() {
    clearTimeout(API.retryTimeout);
    console.log("Trying to connect to the internet!");
    API.retryTimeout = setTimeout(() => API.dequeue(), 5000);
  }

  static afterPromise = (item, resolve = false, data) => {
    this.workingOnPromise = false;
    if (resolve) item.resolve(data);
    else item.reject(data);
    API.dequeue();
  }

  static dequeue() {
    if (this.workingOnPromise) {
      return false;
    }
    if (window.navigator && !window.navigator.onLine) {
      return API.retry();
    }
    const item = API.current_queue.shift();
    if (!item) {
      return false;
    }
    try {
      this.workingOnPromise = true;
      item.promise()
        .then((value) => API.afterPromise(item, true, value))
        .catch((error) => API.afterPromise(item, false, error))
    } catch (error) {
      API.afterPromise(item, false, error);
    }
    return true;
  }

  static debug() {
    return new Promise((res) => setTimeout(
      () => res("Completed")
    , 2000))
  }

  static async request(options) {
    return this.http(options)
      .then((response) => response.data)
      .catch((error) => Promise.reject(error));
  }
}

new API();

const queuedAPI = {
  currentQueue: API.current_queue,
  /* Debug */
  debug() {
    return API.queue(() => API.debug());
  },
  /* Debug */
  get(url = '', options = {}) {
    return API.queue(() => API.request({ method: 'get', url, ...options }));
  },
  post(url = '', options = {}) {
    return API.queue(() => API.request({ method: 'post', url, ...options }));
  },
  put(url = '', options = {}) {
    return API.queue(() => API.request({ method: 'put', url, ...options }));
  },
  delete(url = '', options = {}) {
    return API.queue(() => API.request({ method: 'delete', url, ...options }));
  }
}

export default queuedAPI;
