// ==UserScript==
// @name         NYCU E3 CAPTCHA Autowrite
// @namespace    http://tampermonkey.net/
// @version      1.0
// @description  try to take over the world!
// @author       Chan-Yu Li
// @match        https://e3.nycu.edu.tw/login/index.php
// @icon         https://www.google.com/s2/favicons?domain=tampermonkey.net
// @grant        unsafeWindow
// @grant        GM_xmlhttpRequest
// @connect      127.0.0.1
// ==/UserScript==

(function() {
    'use strict';
    unsafeWindow.onload = function() {
        var canvas = document.createElement("canvas");
        var img = document.getElementById('captcha-desktop'),
            ctx = canvas.getContext('2d');
        document.body.appendChild(canvas);
        canvas.style.display = 'none';
        canvas.height = img.height;
        canvas.width = img.width;
        ctx.drawImage(img, 0, 0);

        var data_url = canvas.toDataURL("image/png");

        //console.log(data_url);

        GM_xmlhttpRequest ( {
            method:     "POST",
            url:        "http://127.0.0.1:5000/e3autologin_base64",
            data:       "file=" + data_url,
            headers:    {
                "Content-Type": "application/x-www-form-urlencoded"
            },
            onload: function (response) {
                document.getElementsByName('captcha_code')[0].value = response.responseText;
            },
            onerror: function () {
                alert('CAPTCHA Autowrite Failed!');
            }
        });

    };

    // Your code here...
})();
