(()=>{var W,l,oe,Ie,C,ee,ie,j,We,F={},ae=[],Be=/acit|ex(?:s|g|n|p|$)|rph|grid|ows|mnc|ntw|ine[ch]|zoo|^ord|itera/i,q=Array.isArray;function S(e,t){for(var n in t)e[n]=t[n];return e}function ue(e){var t=e.parentNode;t&&t.removeChild(e)}function h(e,t,n){var o,i,r,c={};for(r in t)r=="key"?o=t[r]:r=="ref"?i=t[r]:c[r]=t[r];if(arguments.length>2&&(c.children=arguments.length>3?W.call(arguments,2):n),typeof e=="function"&&e.defaultProps!=null)for(r in e.defaultProps)c[r]===void 0&&(c[r]=e.defaultProps[r]);return U(e,c,o,i,null)}function U(e,t,n,o,i){var r={type:e,props:t,key:n,ref:o,__k:null,__:null,__b:0,__e:null,__d:void 0,__c:null,__h:null,constructor:void 0,__v:i??++oe};return i==null&&l.vnode!=null&&l.vnode(r),r}function B(e){return e.children}function D(e,t){this.props=e,this.context=t}function E(e,t){if(t==null)return e.__?E(e.__,e.__.__k.indexOf(e)+1):null;for(var n;t<e.__k.length;t++)if((n=e.__k[t])!=null&&n.__e!=null)return n.__e;return typeof e.type=="function"?E(e):null}function ce(e){var t,n;if((e=e.__)!=null&&e.__c!=null){for(e.__e=e.__c.base=null,t=0;t<e.__k.length;t++)if((n=e.__k[t])!=null&&n.__e!=null){e.__e=e.__c.base=n.__e;break}return ce(e)}}function te(e){(!e.__d&&(e.__d=!0)&&C.push(e)&&!T.__r++||ee!==l.debounceRendering)&&((ee=l.debounceRendering)||ie)(T)}function T(){var e,t,n,o,i,r,c,p;for(C.sort(j);e=C.shift();)e.__d&&(t=C.length,o=void 0,i=void 0,c=(r=(n=e).__v).__e,(p=n.__P)&&(o=[],(i=S({},r)).__v=r.__v+1,O(p,r,i,n.__n,p.ownerSVGElement!==void 0,r.__h!=null?[c]:null,o,c??E(r),r.__h),de(o,r),r.__e!=c&&ce(r)),C.length>t&&C.sort(j));T.__r=0}function le(e,t,n,o,i,r,c,p,f,d){var _,m,u,a,s,w,v,g=o&&o.__k||ae,x=g.length;for(n.__k=[],_=0;_<t.length;_++)if((a=n.__k[_]=(a=t[_])==null||typeof a=="boolean"||typeof a=="function"?null:typeof a=="string"||typeof a=="number"||typeof a=="bigint"?U(null,a,null,null,a):q(a)?U(B,{children:a},null,null,null):a.__b>0?U(a.type,a.props,a.key,a.ref?a.ref:null,a.__v):a)!=null){if(a.__=n,a.__b=n.__b+1,(u=g[_])===null||u&&a.key==u.key&&a.type===u.type)g[_]=void 0;else for(m=0;m<x;m++){if((u=g[m])&&a.key==u.key&&a.type===u.type){g[m]=void 0;break}u=null}O(e,a,u=u||F,i,r,c,p,f,d),s=a.__e,(m=a.ref)&&u.ref!=m&&(v||(v=[]),u.ref&&v.push(u.ref,null,a),v.push(m,a.__c||s,a)),s!=null?(w==null&&(w=s),typeof a.type=="function"&&a.__k===u.__k?a.__d=f=se(a,f,e):f=fe(e,a,u,g,s,f),typeof n.type=="function"&&(n.__d=f)):f&&u.__e==f&&f.parentNode!=e&&(f=E(u))}for(n.__e=w,_=x;_--;)g[_]!=null&&(typeof n.type=="function"&&g[_].__e!=null&&g[_].__e==n.__d&&(n.__d=pe(o).nextSibling),me(g[_],g[_]));if(v)for(_=0;_<v.length;_++)he(v[_],v[++_],v[++_])}function se(e,t,n){for(var o,i=e.__k,r=0;i&&r<i.length;r++)(o=i[r])&&(o.__=e,t=typeof o.type=="function"?se(o,t,n):fe(n,o,o,i,o.__e,t));return t}function fe(e,t,n,o,i,r){var c,p,f;if(t.__d!==void 0)c=t.__d,t.__d=void 0;else if(n==null||i!=r||i.parentNode==null)e:if(r==null||r.parentNode!==e)e.appendChild(i),c=null;else{for(p=r,f=0;(p=p.nextSibling)&&f<o.length;f+=1)if(p==i)break e;e.insertBefore(i,r),c=r}return c!==void 0?c:i.nextSibling}function pe(e){var t,n,o;if(e.type==null||typeof e.type=="string")return e.__e;if(e.__k){for(t=e.__k.length-1;t>=0;t--)if((n=e.__k[t])&&(o=pe(n)))return o}return null}function Re(e,t,n,o,i){var r;for(r in n)r==="children"||r==="key"||r in t||I(e,r,null,n[r],o);for(r in t)i&&typeof t[r]!="function"||r==="children"||r==="key"||r==="value"||r==="checked"||n[r]===t[r]||I(e,r,t[r],n[r],o)}function ne(e,t,n){t[0]==="-"?e.setProperty(t,n??""):e[t]=n==null?"":typeof n!="number"||Be.test(t)?n:n+"px"}function I(e,t,n,o,i){var r;e:if(t==="style")if(typeof n=="string")e.style.cssText=n;else{if(typeof o=="string"&&(e.style.cssText=o=""),o)for(t in o)n&&t in n||ne(e.style,t,"");if(n)for(t in n)o&&n[t]===o[t]||ne(e.style,t,n[t])}else if(t[0]==="o"&&t[1]==="n")r=t!==(t=t.replace(/Capture$/,"")),t=t.toLowerCase()in e?t.toLowerCase().slice(2):t.slice(2),e.l||(e.l={}),e.l[t+r]=n,n?o||e.addEventListener(t,r?re:_e,r):e.removeEventListener(t,r?re:_e,r);else if(t!=="dangerouslySetInnerHTML"){if(i)t=t.replace(/xlink(H|:h)/,"h").replace(/sName$/,"s");else if(t!=="width"&&t!=="height"&&t!=="href"&&t!=="list"&&t!=="form"&&t!=="tabIndex"&&t!=="download"&&t!=="rowSpan"&&t!=="colSpan"&&t in e)try{e[t]=n??"";break e}catch{}typeof n=="function"||(n==null||n===!1&&t[4]!=="-"?e.removeAttribute(t):e.setAttribute(t,n))}}function _e(e){return this.l[e.type+!1](l.event?l.event(e):e)}function re(e){return this.l[e.type+!0](l.event?l.event(e):e)}function O(e,t,n,o,i,r,c,p,f){var d,_,m,u,a,s,w,v,g,x,H,A,Z,$,V,k=t.type;if(t.constructor!==void 0)return null;n.__h!=null&&(f=n.__h,p=t.__e=n.__e,t.__h=null,r=[p]),(d=l.__b)&&d(t);try{e:if(typeof k=="function"){if(v=t.props,g=(d=k.contextType)&&o[d.__c],x=d?g?g.props.value:d.__:o,n.__c?w=(_=t.__c=n.__c).__=_.__E:("prototype"in k&&k.prototype.render?t.__c=_=new k(v,x):(t.__c=_=new D(v,x),_.constructor=k,_.render=Me),g&&g.sub(_),_.props=v,_.state||(_.state={}),_.context=x,_.__n=o,m=_.__d=!0,_.__h=[],_._sb=[]),_.__s==null&&(_.__s=_.state),k.getDerivedStateFromProps!=null&&(_.__s==_.state&&(_.__s=S({},_.__s)),S(_.__s,k.getDerivedStateFromProps(v,_.__s))),u=_.props,a=_.state,_.__v=t,m)k.getDerivedStateFromProps==null&&_.componentWillMount!=null&&_.componentWillMount(),_.componentDidMount!=null&&_.__h.push(_.componentDidMount);else{if(k.getDerivedStateFromProps==null&&v!==u&&_.componentWillReceiveProps!=null&&_.componentWillReceiveProps(v,x),!_.__e&&_.shouldComponentUpdate!=null&&_.shouldComponentUpdate(v,_.__s,x)===!1||t.__v===n.__v){for(t.__v!==n.__v&&(_.props=v,_.state=_.__s,_.__d=!1),_.__e=!1,t.__e=n.__e,t.__k=n.__k,t.__k.forEach(function(N){N&&(N.__=t)}),H=0;H<_._sb.length;H++)_.__h.push(_._sb[H]);_._sb=[],_.__h.length&&c.push(_);break e}_.componentWillUpdate!=null&&_.componentWillUpdate(v,_.__s,x),_.componentDidUpdate!=null&&_.__h.push(function(){_.componentDidUpdate(u,a,s)})}if(_.context=x,_.props=v,_.__P=e,A=l.__r,Z=0,"prototype"in k&&k.prototype.render){for(_.state=_.__s,_.__d=!1,A&&A(t),d=_.render(_.props,_.state,_.context),$=0;$<_._sb.length;$++)_.__h.push(_._sb[$]);_._sb=[]}else do _.__d=!1,A&&A(t),d=_.render(_.props,_.state,_.context),_.state=_.__s;while(_.__d&&++Z<25);_.state=_.__s,_.getChildContext!=null&&(o=S(S({},o),_.getChildContext())),m||_.getSnapshotBeforeUpdate==null||(s=_.getSnapshotBeforeUpdate(u,a)),le(e,q(V=d!=null&&d.type===B&&d.key==null?d.props.children:d)?V:[V],t,n,o,i,r,c,p,f),_.base=t.__e,t.__h=null,_.__h.length&&c.push(_),w&&(_.__E=_.__=null),_.__e=!1}else r==null&&t.__v===n.__v?(t.__k=n.__k,t.__e=n.__e):t.__e=Le(n.__e,t,n,o,i,r,c,f);(d=l.diffed)&&d(t)}catch(N){t.__v=null,(f||r!=null)&&(t.__e=p,t.__h=!!f,r[r.indexOf(p)]=null),l.__e(N,t,n)}}function de(e,t){l.__c&&l.__c(t,e),e.some(function(n){try{e=n.__h,n.__h=[],e.some(function(o){o.call(n)})}catch(o){l.__e(o,n.__v)}})}function Le(e,t,n,o,i,r,c,p){var f,d,_,m=n.props,u=t.props,a=t.type,s=0;if(a==="svg"&&(i=!0),r!=null){for(;s<r.length;s++)if((f=r[s])&&"setAttribute"in f==!!a&&(a?f.localName===a:f.nodeType===3)){e=f,r[s]=null;break}}if(e==null){if(a===null)return document.createTextNode(u);e=i?document.createElementNS("http://www.w3.org/2000/svg",a):document.createElement(a,u.is&&u),r=null,p=!1}if(a===null)m===u||p&&e.data===u||(e.data=u);else{if(r=r&&W.call(e.childNodes),d=(m=n.props||F).dangerouslySetInnerHTML,_=u.dangerouslySetInnerHTML,!p){if(r!=null)for(m={},s=0;s<e.attributes.length;s++)m[e.attributes[s].name]=e.attributes[s].value;(_||d)&&(_&&(d&&_.__html==d.__html||_.__html===e.innerHTML)||(e.innerHTML=_&&_.__html||""))}if(Re(e,u,m,i,p),_)t.__k=[];else if(le(e,q(s=t.props.children)?s:[s],t,n,o,i&&a!=="foreignObject",r,c,r?r[0]:n.__k&&E(n,0),p),r!=null)for(s=r.length;s--;)r[s]!=null&&ue(r[s]);p||("value"in u&&(s=u.value)!==void 0&&(s!==e.value||a==="progress"&&!s||a==="option"&&s!==m.value)&&I(e,"value",s,m.value,!1),"checked"in u&&(s=u.checked)!==void 0&&s!==e.checked&&I(e,"checked",s,m.checked,!1))}return e}function he(e,t,n){try{typeof e=="function"?e(t):e.current=t}catch(o){l.__e(o,n)}}function me(e,t,n){var o,i;if(l.unmount&&l.unmount(e),(o=e.ref)&&(o.current&&o.current!==e.__e||he(o,null,t)),(o=e.__c)!=null){if(o.componentWillUnmount)try{o.componentWillUnmount()}catch(r){l.__e(r,t)}o.base=o.__P=null,e.__c=void 0}if(o=e.__k)for(i=0;i<o.length;i++)o[i]&&me(o[i],t,n||typeof e.type!="function");n||e.__e==null||ue(e.__e),e.__=e.__e=e.__d=void 0}function Me(e,t,n){return this.constructor(e,n)}function ve(e,t,n){var o,i,r;l.__&&l.__(e,t),i=(o=typeof n=="function")?null:n&&n.__k||t.__k,r=[],O(t,e=(!o&&n||t).__k=h(B,null,[e]),i||F,F,t.ownerSVGElement!==void 0,!o&&n?[n]:i?null:t.firstChild?W.call(t.childNodes):null,r,!o&&n?n:i?i.__e:t.firstChild,o),de(r,e)}W=ae.slice,l={__e:function(e,t,n,o){for(var i,r,c;t=t.__;)if((i=t.__c)&&!i.__)try{if((r=i.constructor)&&r.getDerivedStateFromError!=null&&(i.setState(r.getDerivedStateFromError(e)),c=i.__d),i.componentDidCatch!=null&&(i.componentDidCatch(e,o||{}),c=i.__d),c)return i.__E=i}catch(p){e=p}throw e}},oe=0,Ie=function(e){return e!=null&&e.constructor===void 0},D.prototype.setState=function(e,t){var n;n=this.__s!=null&&this.__s!==this.state?this.__s:this.__s=S({},this.state),typeof e=="function"&&(e=e(S({},n),this.props)),e&&S(n,e),e!=null&&this.__v&&(t&&this._sb.push(t),te(this))},D.prototype.forceUpdate=function(e){this.__v&&(this.__e=!0,e&&this.__h.push(e),te(this))},D.prototype.render=B,C=[],ie=typeof Promise=="function"?Promise.prototype.then.bind(Promise.resolve()):setTimeout,j=function(e,t){return e.__v.__b-t.__v.__b},T.__r=0,We=0;var z=[{name:"LinkedIn",sitekey:"3117BF26-4762-4F5A-8ED9-A85E69209A46",loader:1},{name:"Blizzard",sitekey:"E8A75615-1CBA-5DFF-8032-D16BCF234E10"},{name:"Discovery+",sitekey:"FE296399-FDEA-2EA2-8CD5-50F6E3157ECA"}];var M,y,G,ye,J=0,we=[],R=[],ge=l.__b,be=l.__r,ke=l.diffed,xe=l.__c,Se=l.unmount;function Ae(e,t){l.__h&&l.__h(y,e,J||t),J=0;var n=y.__H||(y.__H={__:[],__h:[]});return e>=n.__.length&&n.__.push({__V:R}),n.__[e]}function P(e){return J=1,Ve(Pe,e)}function Ve(e,t,n){var o=Ae(M++,2);if(o.t=e,!o.__c&&(o.__=[n?n(t):Pe(void 0,t),function(p){var f=o.__N?o.__N[0]:o.__[0],d=o.t(f,p);f!==d&&(o.__N=[d,o.__[1]],o.__c.setState({}))}],o.__c=y,!y.u)){var i=function(p,f,d){if(!o.__c.__H)return!0;var _=o.__c.__H.__.filter(function(u){return u.__c});if(_.every(function(u){return!u.__N}))return!r||r.call(this,p,f,d);var m=!1;return _.forEach(function(u){if(u.__N){var a=u.__[0];u.__=u.__N,u.__N=void 0,a!==u.__[0]&&(m=!0)}}),!(!m&&o.__c.props===p)&&(!r||r.call(this,p,f,d))};y.u=!0;var r=y.shouldComponentUpdate,c=y.componentWillUpdate;y.componentWillUpdate=function(p,f,d){if(this.__e){var _=r;r=void 0,i(p,f,d),r=_}c&&c.call(this,p,f,d)},y.shouldComponentUpdate=i}return o.__N||o.__}function Ee(e,t){var n=Ae(M++,3);!l.__s&&Oe(n.__H,t)&&(n.__=e,n.i=t,y.__H.__h.push(n))}function je(){for(var e;e=we.shift();)if(e.__P&&e.__H)try{e.__H.__h.forEach(L),e.__H.__h.forEach(K),e.__H.__h=[]}catch(t){e.__H.__h=[],l.__e(t,e.__v)}}l.__b=function(e){y=null,ge&&ge(e)},l.__r=function(e){be&&be(e),M=0;var t=(y=e.__c).__H;t&&(G===y?(t.__h=[],y.__h=[],t.__.forEach(function(n){n.__N&&(n.__=n.__N),n.__V=R,n.__N=n.i=void 0})):(t.__h.forEach(L),t.__h.forEach(K),t.__h=[],M=0)),G=y},l.diffed=function(e){ke&&ke(e);var t=e.__c;t&&t.__H&&(t.__H.__h.length&&(we.push(t)!==1&&ye===l.requestAnimationFrame||((ye=l.requestAnimationFrame)||qe)(je)),t.__H.__.forEach(function(n){n.i&&(n.__H=n.i),n.__V!==R&&(n.__=n.__V),n.i=void 0,n.__V=R})),G=y=null},l.__c=function(e,t){t.some(function(n){try{n.__h.forEach(L),n.__h=n.__h.filter(function(o){return!o.__||K(o)})}catch(o){t.some(function(i){i.__h&&(i.__h=[])}),t=[],l.__e(o,n.__v)}}),xe&&xe(e,t)},l.unmount=function(e){Se&&Se(e);var t,n=e.__c;n&&n.__H&&(n.__H.__.forEach(function(o){try{L(o)}catch(i){t=i}}),n.__H=void 0,t&&l.__e(t,n.__v))};var Ce=typeof requestAnimationFrame=="function";function qe(e){var t,n=function(){clearTimeout(o),Ce&&cancelAnimationFrame(t),setTimeout(e)},o=setTimeout(n,100);Ce&&(t=requestAnimationFrame(n))}function L(e){var t=y,n=e.__c;typeof n=="function"&&(e.__c=void 0,n()),y=t}function K(e){var t=y;e.__c=e.__(),y=t}function Oe(e,t){return!e||e.length!==t.length||t.some(function(n,o){return n!==e[o]})}function Pe(e,t){return typeof t=="function"?t(e):t}var $e=chrome;var Q="https://api.nopecha.com",b="https://www.nopecha.com",He="https://developers.nopecha.com",Xe={doc:{url:He,automation:{url:`${He}/guides/extension_advanced/#automation-build`}},api:{url:Q,recognition:{url:`${Q}/recognition`},status:{url:`${Q}/status`}},www:{url:b,annoucement:{url:`${b}/json/announcement.json`},demo:{url:`${b}/captcha`,recaptcha:{url:`${b}/captcha/recaptcha`},funcaptcha:{url:`${b}/captcha/funcaptcha`},awscaptcha:{url:`${b}/captcha/awscaptcha`},textcaptcha:{url:`${b}/captcha/textcaptcha`},turnstile:{url:`${b}/captcha/turnstile`},perimeterx:{url:`${b}/captcha/perimeterx`},geetest:{url:`${b}/captcha/geetest`},lemincaptcha:{url:`${b}/captcha/lemincaptcha`}},manage:{url:`${b}/manage`},pricing:{url:`${b}/pricing`},setup:{url:`${b}/setup`}},discord:{url:`${b}/discord`},github:{url:`${b}/github`,release:{url:`${b}/github/release`}}};function Ne(e){let t=("60a8b3778b5b01f87ccc8129cd88bf0f6ec61feb879c88908365771cfcadc232"+e).split("").map(n=>n.charCodeAt(0));return De(t)}var Ue=new Uint32Array(256);for(let e=256;e--;){let t=e;for(let n=8;n--;)t=t&1?3988292384^t>>>1:t>>>1;Ue[e]=t}function De(e){let t=-1;for(let n of e)t=t>>>8^Ue[t&255^n];return(t^-1)>>>0}async function Fe(e,t){let n=""+[+new Date,performance.now(),Math.random()],[o,i]=await new Promise(r=>{$e.runtime.sendMessage([n,e,...t],r)});if(o===Ne(n))return i}function Y(){let[e,t]=P(1),[n,o]=P(!1),[i,r]=P(null);return h("main",{style:{maxWidth:"100rem",padding:"1rem",margin:"0px auto",fontFamily:'ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji"'}},h(X,null,h("h4",null,"Choose a FunCAPTCHA sitekey"),h("div",{style:{display:"flex",flexWrap:"wrap",gap:"0.125rem 0.25rem"}},h("button",{onClick:()=>{o(!n),r(null)}},"Show all"),z.map(c=>h("button",{onClick:()=>{o(!1),r(c)}},c.name)))),h(X,null,h("h4",null,"Number of copies to render"),h("div",{style:{display:"flex"}},h("input",{type:"range",min:1,max:20,value:e,onChange:c=>{t(c.currentTarget.valueAsNumber)},style:{width:"300px",maxWidth:"80%"}}),e)),h(X,null,!n&&!i?h("p",null):n?z.map(c=>h(Te,{sitekey:c,amount:e})):h(Te,{sitekey:i,amount:e})))}function X({children:e}){return h("section",{style:{margin:"2rem 0px"}},e)}function Te({sitekey:e,amount:t}){return h("div",{style:{margin:"0.25rem 0px"}},h("h4",null,e.name),h("hr",null),h("div",{style:{display:"flex",flexWrap:"wrap"}},Array(t).fill(0).map((n,o)=>h(ze,{...e}))))}function ze({sitekey:e,hostname:t="iframe.arkoselabs.com",dimensions:[n,o]=[320,310],loader:i=0}){let[r,c]=P();return Ee(()=>{if(i===0)c(`https://${t}/${e}/index.html?mkt=en`);else if(i===1){let p=`https://api.funcaptcha.com/fc/gt2/public_key/${e}`;Fe("fetch::universalFetch",[p,{method:"POST",body:new URLSearchParams({bda:"",site:"",public_key:e,language:"en",userbrowser:navigator.userAgent,rnd:""+Math.random()}).toString(),headers:{"content-type":"application/x-www-form-urlencoded; charset=UTF-8"}}]).then(({text:f})=>{let d=JSON.parse(f),_={};for(let u of d.token.split("|")){let[a,s]=u.split("=");s||([a,s]=["token",a]),a.endsWith("url")&&(s=decodeURIComponent(s)),_[a]=s}let m=new URLSearchParams(_);c(`https://api.funcaptcha.com/fc/gc/?${m.toString()}`)})}},[e,t,i]),h("div",{style:{width:`${n}px`,height:`${o}px`,border:"1px solid black"}},r?h("iframe",{src:r,style:{width:"100%",height:"100%",border:"none"}}):h("p",null,"Loading..."))}[...document.body.children].forEach(e=>e.remove());ve(h(Y,{}),document.body);})();