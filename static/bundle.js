!function(t){var e={};function r(n){if(e[n])return e[n].exports;var o=e[n]={i:n,l:!1,exports:{}};return t[n].call(o.exports,o,o.exports,r),o.l=!0,o.exports}r.m=t,r.c=e,r.d=function(t,e,n){r.o(t,e)||Object.defineProperty(t,e,{enumerable:!0,get:n})},r.r=function(t){"undefined"!=typeof Symbol&&Symbol.toStringTag&&Object.defineProperty(t,Symbol.toStringTag,{value:"Module"}),Object.defineProperty(t,"__esModule",{value:!0})},r.t=function(t,e){if(1&e&&(t=r(t)),8&e)return t;if(4&e&&"object"==typeof t&&t&&t.__esModule)return t;var n=Object.create(null);if(r.r(n),Object.defineProperty(n,"default",{enumerable:!0,value:t}),2&e&&"string"!=typeof t)for(var o in t)r.d(n,o,function(e){return t[e]}.bind(null,o));return n},r.n=function(t){var e=t&&t.__esModule?function(){return t.default}:function(){return t};return r.d(e,"a",e),e},r.o=function(t,e){return Object.prototype.hasOwnProperty.call(t,e)},r.p="",r(r.s=0)}([function(t,e,r){r(2),t.exports=r(1)},function(t,e,r){},function(t,e,r){"use strict";r.r(e);var n="undefined"!=typeof globalThis&&globalThis||"undefined"!=typeof self&&self||void 0!==n&&n,o="URLSearchParams"in n,i="Symbol"in n&&"iterator"in Symbol,s="FileReader"in n&&"Blob"in n&&function(){try{return new Blob,!0}catch(t){return!1}}(),a="FormData"in n,u="ArrayBuffer"in n;if(u)var d=["[object Int8Array]","[object Uint8Array]","[object Uint8ClampedArray]","[object Int16Array]","[object Uint16Array]","[object Int32Array]","[object Uint32Array]","[object Float32Array]","[object Float64Array]"],c=ArrayBuffer.isView||function(t){return t&&d.indexOf(Object.prototype.toString.call(t))>-1};function f(t){if("string"!=typeof t&&(t=String(t)),/[^a-z0-9\-#$%&'*+.^_`|~!]/i.test(t)||""===t)throw new TypeError("Invalid character in header field name");return t.toLowerCase()}function l(t){return"string"!=typeof t&&(t=String(t)),t}function h(t){var e={next:function(){var e=t.shift();return{done:void 0===e,value:e}}};return i&&(e[Symbol.iterator]=function(){return e}),e}function y(t){this.map={},t instanceof y?t.forEach((function(t,e){this.append(e,t)}),this):Array.isArray(t)?t.forEach((function(t){this.append(t[0],t[1])}),this):t&&Object.getOwnPropertyNames(t).forEach((function(e){this.append(e,t[e])}),this)}function p(t){if(t.bodyUsed)return Promise.reject(new TypeError("Already read"));t.bodyUsed=!0}function b(t){return new Promise((function(e,r){t.onload=function(){e(t.result)},t.onerror=function(){r(t.error)}}))}function m(t){var e=new FileReader,r=b(e);return e.readAsArrayBuffer(t),r}function w(t){if(t.slice)return t.slice(0);var e=new Uint8Array(t.byteLength);return e.set(new Uint8Array(t)),e.buffer}function v(){return this.bodyUsed=!1,this._initBody=function(t){var e;this.bodyUsed=this.bodyUsed,this._bodyInit=t,t?"string"==typeof t?this._bodyText=t:s&&Blob.prototype.isPrototypeOf(t)?this._bodyBlob=t:a&&FormData.prototype.isPrototypeOf(t)?this._bodyFormData=t:o&&URLSearchParams.prototype.isPrototypeOf(t)?this._bodyText=t.toString():u&&s&&((e=t)&&DataView.prototype.isPrototypeOf(e))?(this._bodyArrayBuffer=w(t.buffer),this._bodyInit=new Blob([this._bodyArrayBuffer])):u&&(ArrayBuffer.prototype.isPrototypeOf(t)||c(t))?this._bodyArrayBuffer=w(t):this._bodyText=t=Object.prototype.toString.call(t):this._bodyText="",this.headers.get("content-type")||("string"==typeof t?this.headers.set("content-type","text/plain;charset=UTF-8"):this._bodyBlob&&this._bodyBlob.type?this.headers.set("content-type",this._bodyBlob.type):o&&URLSearchParams.prototype.isPrototypeOf(t)&&this.headers.set("content-type","application/x-www-form-urlencoded;charset=UTF-8"))},s&&(this.blob=function(){var t=p(this);if(t)return t;if(this._bodyBlob)return Promise.resolve(this._bodyBlob);if(this._bodyArrayBuffer)return Promise.resolve(new Blob([this._bodyArrayBuffer]));if(this._bodyFormData)throw new Error("could not read FormData body as blob");return Promise.resolve(new Blob([this._bodyText]))},this.arrayBuffer=function(){if(this._bodyArrayBuffer){var t=p(this);return t||(ArrayBuffer.isView(this._bodyArrayBuffer)?Promise.resolve(this._bodyArrayBuffer.buffer.slice(this._bodyArrayBuffer.byteOffset,this._bodyArrayBuffer.byteOffset+this._bodyArrayBuffer.byteLength)):Promise.resolve(this._bodyArrayBuffer))}return this.blob().then(m)}),this.text=function(){var t,e,r,n=p(this);if(n)return n;if(this._bodyBlob)return t=this._bodyBlob,e=new FileReader,r=b(e),e.readAsText(t),r;if(this._bodyArrayBuffer)return Promise.resolve(function(t){for(var e=new Uint8Array(t),r=new Array(e.length),n=0;n<e.length;n++)r[n]=String.fromCharCode(e[n]);return r.join("")}(this._bodyArrayBuffer));if(this._bodyFormData)throw new Error("could not read FormData body as text");return Promise.resolve(this._bodyText)},a&&(this.formData=function(){return this.text().then(A)}),this.json=function(){return this.text().then(JSON.parse)},this}y.prototype.append=function(t,e){t=f(t),e=l(e);var r=this.map[t];this.map[t]=r?r+", "+e:e},y.prototype.delete=function(t){delete this.map[f(t)]},y.prototype.get=function(t){return t=f(t),this.has(t)?this.map[t]:null},y.prototype.has=function(t){return this.map.hasOwnProperty(f(t))},y.prototype.set=function(t,e){this.map[f(t)]=l(e)},y.prototype.forEach=function(t,e){for(var r in this.map)this.map.hasOwnProperty(r)&&t.call(e,this.map[r],r,this)},y.prototype.keys=function(){var t=[];return this.forEach((function(e,r){t.push(r)})),h(t)},y.prototype.values=function(){var t=[];return this.forEach((function(e){t.push(e)})),h(t)},y.prototype.entries=function(){var t=[];return this.forEach((function(e,r){t.push([r,e])})),h(t)},i&&(y.prototype[Symbol.iterator]=y.prototype.entries);var g=["DELETE","GET","HEAD","OPTIONS","POST","PUT"];function _(t,e){if(!(this instanceof _))throw new TypeError('Please use the "new" operator, this DOM object constructor cannot be called as a function.');var r,n,o=(e=e||{}).body;if(t instanceof _){if(t.bodyUsed)throw new TypeError("Already read");this.url=t.url,this.credentials=t.credentials,e.headers||(this.headers=new y(t.headers)),this.method=t.method,this.mode=t.mode,this.signal=t.signal,o||null==t._bodyInit||(o=t._bodyInit,t.bodyUsed=!0)}else this.url=String(t);if(this.credentials=e.credentials||this.credentials||"same-origin",!e.headers&&this.headers||(this.headers=new y(e.headers)),this.method=(r=e.method||this.method||"GET",n=r.toUpperCase(),g.indexOf(n)>-1?n:r),this.mode=e.mode||this.mode||null,this.signal=e.signal||this.signal,this.referrer=null,("GET"===this.method||"HEAD"===this.method)&&o)throw new TypeError("Body not allowed for GET or HEAD requests");if(this._initBody(o),!("GET"!==this.method&&"HEAD"!==this.method||"no-store"!==e.cache&&"no-cache"!==e.cache)){var i=/([?&])_=[^&]*/;if(i.test(this.url))this.url=this.url.replace(i,"$1_="+(new Date).getTime());else{this.url+=(/\?/.test(this.url)?"&":"?")+"_="+(new Date).getTime()}}}function A(t){var e=new FormData;return t.trim().split("&").forEach((function(t){if(t){var r=t.split("="),n=r.shift().replace(/\+/g," "),o=r.join("=").replace(/\+/g," ");e.append(decodeURIComponent(n),decodeURIComponent(o))}})),e}function E(t,e){if(!(this instanceof E))throw new TypeError('Please use the "new" operator, this DOM object constructor cannot be called as a function.');e||(e={}),this.type="default",this.status=void 0===e.status?200:e.status,this.ok=this.status>=200&&this.status<300,this.statusText="statusText"in e?e.statusText:"",this.headers=new y(e.headers),this.url=e.url||"",this._initBody(t)}_.prototype.clone=function(){return new _(this,{body:this._bodyInit})},v.call(_.prototype),v.call(E.prototype),E.prototype.clone=function(){return new E(this._bodyInit,{status:this.status,statusText:this.statusText,headers:new y(this.headers),url:this.url})},E.error=function(){var t=new E(null,{status:0,statusText:""});return t.type="error",t};var T=[301,302,303,307,308];E.redirect=function(t,e){if(-1===T.indexOf(e))throw new RangeError("Invalid status code");return new E(null,{status:e,headers:{location:t}})};var B=n.DOMException;try{new B}catch(t){(B=function(t,e){this.message=t,this.name=e;var r=Error(t);this.stack=r.stack}).prototype=Object.create(Error.prototype),B.prototype.constructor=B}function x(t,e){return new Promise((function(r,o){var i=new _(t,e);if(i.signal&&i.signal.aborted)return o(new B("Aborted","AbortError"));var a=new XMLHttpRequest;function d(){a.abort()}a.onload=function(){var t,e,n={status:a.status,statusText:a.statusText,headers:(t=a.getAllResponseHeaders()||"",e=new y,t.replace(/\r?\n[\t ]+/g," ").split("\r").map((function(t){return 0===t.indexOf("\n")?t.substr(1,t.length):t})).forEach((function(t){var r=t.split(":"),n=r.shift().trim();if(n){var o=r.join(":").trim();e.append(n,o)}})),e)};n.url="responseURL"in a?a.responseURL:n.headers.get("X-Request-URL");var o="response"in a?a.response:a.responseText;setTimeout((function(){r(new E(o,n))}),0)},a.onerror=function(){setTimeout((function(){o(new TypeError("Network request failed"))}),0)},a.ontimeout=function(){setTimeout((function(){o(new TypeError("Network request failed"))}),0)},a.onabort=function(){setTimeout((function(){o(new B("Aborted","AbortError"))}),0)},a.open(i.method,function(t){try{return""===t&&n.location.href?n.location.href:t}catch(e){return t}}(i.url),!0),"include"===i.credentials?a.withCredentials=!0:"omit"===i.credentials&&(a.withCredentials=!1),"responseType"in a&&(s?a.responseType="blob":u&&i.headers.get("Content-Type")&&-1!==i.headers.get("Content-Type").indexOf("application/octet-stream")&&(a.responseType="arraybuffer")),!e||"object"!=typeof e.headers||e.headers instanceof y?i.headers.forEach((function(t,e){a.setRequestHeader(e,t)})):Object.getOwnPropertyNames(e.headers).forEach((function(t){a.setRequestHeader(t,l(e.headers[t]))})),i.signal&&(i.signal.addEventListener("abort",d),a.onreadystatechange=function(){4===a.readyState&&i.signal.removeEventListener("abort",d)}),a.send(void 0===i._bodyInit?null:i._bodyInit)}))}x.polyfill=!0,n.fetch||(n.fetch=x,n.Headers=y,n.Request=_,n.Response=E),window.addEventListener("load",(function(t){var e=new Image;e.src=currentContentItem.next_item_url;var r=document.getElementById("image-payload"),n=document.getElementById("transition-video-payload");n.load();var o=document.getElementById("next-button"),i=document.getElementById("display-link"),s=document.getElementById("copy-a"),a=document.getElementById("tweet-a"),u=document.getElementById("asset-crypto-url"),d=document.getElementById("asset-crypto-section");function c(t){currentContentItem=t,r.setAttribute("src",t.url),e.src=t.next_item_url,n.src=t.transition_url,n.load(),s.setAttribute("href",t.permalink),a.setAttribute("href","https://twitter.com/intent/tweet?url="+encodeURIComponent(s.href)),t.opensea_item_url?(d.style.display="",u.setAttribute("href",t.opensea_item_url)):(d.style.display="none",u.setAttribute("href","#"))}function f(){r.setAttribute("src",e.src),r.style.zIndex="1",n.style.zIndex="0",setTimeout((function(){window.fetch("/item/"+currentContentItem.next_item_key).then((function(t){return t.json()})).then((function(t){if(!t.url)return console.error("Bad response!"),o.disabled=!1,void o.classList.remove("disabled");c(t),history.pushState({contentItem:currentContentItem},"",isRootRequest?void 0:currentContentItem.permalink),o.disabled=!1,o.classList.remove("disabled")})).catch((function(t){console.log(t),o.disabled=!1,o.classList.remove("disabled")}))}),50)}function l(){r.style.zIndex="0",n.style.zIndex="1",o.disabled=!0,o.classList.add("disabled");var t=[0];!function e(){4!=n.readyState&&t[0]<1e3?(t[0]+=100,setTimeout(e,100)):4!=n.readyState?f():(n.play(),n.onended=function(t){f()})}()}s.setAttribute("href",currentContentItem.permalink);var h=!0,y=setInterval((function(){!o.disabled&&document.hasFocus&&h&&l()}),7500);function p(){h=!1,clearInterval(y),o.disabled||l()}o.addEventListener("click",(function(t){t.preventDefault(),p()})),i.addEventListener("click",(function(t){t.preventDefault(),p()})),window.addEventListener("popstate",(function(t){console.log(t.state.contentItem),c(t.state.contentItem)}))}))}]);