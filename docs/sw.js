/**
 * Copyright 2018 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// If the loader is already loaded, just stop.
if (!self.define) {
  let registry = {};

  // Used for `eval` and `importScripts` where we can't get script URL by other means.
  // In both cases, it's safe to use a global var because those functions are synchronous.
  let nextDefineUri;

  const singleRequire = (uri, parentUri) => {
    uri = new URL(uri + ".js", parentUri).href;
    return registry[uri] || (
      
        new Promise(resolve => {
          if ("document" in self) {
            const script = document.createElement("script");
            script.src = uri;
            script.onload = resolve;
            document.head.appendChild(script);
          } else {
            nextDefineUri = uri;
            importScripts(uri);
            resolve();
          }
        })
      
      .then(() => {
        let promise = registry[uri];
        if (!promise) {
          throw new Error(`Module ${uri} didn’t register its module`);
        }
        return promise;
      })
    );
  };

  self.define = (depsNames, factory) => {
    const uri = nextDefineUri || ("document" in self ? document.currentScript.src : "") || location.href;
    if (registry[uri]) {
      // Module is already loading or loaded.
      return;
    }
    let exports = {};
    const require = depUri => singleRequire(depUri, uri);
    const specialDeps = {
      module: { uri },
      exports,
      require
    };
    registry[uri] = Promise.all(depsNames.map(
      depName => specialDeps[depName] || require(depName)
    )).then(deps => {
      factory(...deps);
      return exports;
    });
  };
}
define(['./workbox-7e5eb42b'], (function (workbox) { 'use strict';

  self.skipWaiting();
  workbox.clientsClaim();
  /**
   * The precacheAndRoute() method efficiently caches and responds to
   * requests for URLs in the manifest.
   * See https://goo.gl/S9QRab
   */
  workbox.precacheAndRoute([{
    "url": "registerSW.js",
    "revision": "402b66900e731ca748771b6fc5e7a068"
  }, {
    "url": "pwa-512x512.png",
    "revision": "f829b914fc47cfc9c0747c119c27cf1b"
  }, {
    "url": "pwa-192x192.png",
    "revision": "f829b914fc47cfc9c0747c119c27cf1b"
  }, {
    "url": "index.html",
    "revision": "75ac38a9a9f36a06dd5faf90724b410e"
  }, {
    "url": "icon.svg",
    "revision": "1950814dd88b9c7f3f1ff457fc660263"
  }, {
    "url": "apple-touch-icon.png",
    "revision": "f829b914fc47cfc9c0747c119c27cf1b"
  }, {
    "url": "assets/index-C8SzB0Hd.js",
    "revision": null
  }, {
    "url": "assets/index-BkxPOLt6.css",
    "revision": null
  }, {
    "url": "assets/aistudio/image_d731e454.png",
    "revision": null
  }, {
    "url": "apple-touch-icon.png",
    "revision": "f829b914fc47cfc9c0747c119c27cf1b"
  }, {
    "url": "icon.svg",
    "revision": "1950814dd88b9c7f3f1ff457fc660263"
  }, {
    "url": "pwa-192x192.png",
    "revision": "f829b914fc47cfc9c0747c119c27cf1b"
  }, {
    "url": "pwa-512x512.png",
    "revision": "f829b914fc47cfc9c0747c119c27cf1b"
  }, {
    "url": "manifest.webmanifest",
    "revision": "06da87541606934a091273806d71c8b8"
  }], {});
  workbox.cleanupOutdatedCaches();
  workbox.registerRoute(new workbox.NavigationRoute(workbox.createHandlerBoundToURL("index.html")));

}));
