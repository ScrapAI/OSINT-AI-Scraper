(function(){"use strict";const t=t=>Math.floor(Math.random()*Number.MAX_SAFE_INTEGER).toString(16).slice(-t).padStart(t,"0");const n=`${t(8)}${t(8)}${t(8)}${t(8)}`;const e=function(){};e.get=function(t,n){if(!n)n=t;setTimeout((()=>{n([])}),1)};e.getPromise=function(){return Promise.resolve([])};e.getV18=function(){return n};e.x64hash128=function(){return n};e.prototype={get:function(t,e){if(!e)e=t;setTimeout((()=>{e(n,[])}),1)}};self.Fingerprint2=e})();