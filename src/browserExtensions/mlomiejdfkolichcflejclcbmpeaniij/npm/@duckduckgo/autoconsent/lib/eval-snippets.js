const snippets = {
  // code-based rules
  EVAL_0: () => console.log(1),
  EVAL_CONSENTMANAGER_1: () => window.__cmp && typeof __cmp("getCMPData") === "object",
  EVAL_CONSENTMANAGER_2: () => !__cmp("consentStatus").userChoiceExists,
  EVAL_CONSENTMANAGER_3: () => __cmp("setConsent", 0),
  EVAL_CONSENTMANAGER_4: () => __cmp("setConsent", 1),
  EVAL_CONSENTMANAGER_5: () => __cmp("consentStatus").userChoiceExists,
  EVAL_COOKIEBOT_1: () => !!window.Cookiebot,
  EVAL_COOKIEBOT_2: () => !window.Cookiebot.hasResponse && window.Cookiebot.dialog?.visible === true,
  EVAL_COOKIEBOT_3: () => window.Cookiebot.withdraw() || true,
  EVAL_COOKIEBOT_4: () => window.Cookiebot.hide() || true,
  EVAL_COOKIEBOT_5: () => window.Cookiebot.declined === true,
  EVAL_KLARO_1: () => {
    const config = globalThis.klaroConfig || globalThis.klaro?.getManager && globalThis.klaro.getManager().config;
    if (!config) {
      return true;
    }
    const optionalServices = (config.services || config.apps).filter((s) => !s.required).map((s) => s.name);
    if (klaro && klaro.getManager) {
      const manager = klaro.getManager();
      return optionalServices.every((name) => !manager.consents[name]);
    } else if (klaroConfig && klaroConfig.storageMethod === "cookie") {
      const cookieName = klaroConfig.cookieName || klaroConfig.storageName;
      const consents = JSON.parse(decodeURIComponent(document.cookie.split(";").find((c) => c.trim().startsWith(cookieName)).split("=")[1]));
      return Object.keys(consents).filter((k) => optionalServices.includes(k)).every((k) => consents[k] === false);
    }
  },
  EVAL_KLARO_OPEN_POPUP: () => {
    klaro.show(void 0, true);
  },
  EVAL_KLARO_TRY_API_OPT_OUT: () => {
    if (window.klaro && typeof klaro.show === "function" && typeof klaro.getManager === "function") {
      try {
        klaro.getManager().changeAll(false);
        klaro.getManager().saveAndApplyConsents();
        return true;
      } catch (e) {
        console.warn(e);
        return false;
      }
    }
    return false;
  },
  EVAL_ONETRUST_1: () => window.OnetrustActiveGroups.split(",").filter((s) => s.length > 0).length <= 1,
  EVAL_TRUSTARC_TOP: () => window && window.truste && window.truste.eu.bindMap.prefCookie === "0",
  EVAL_TRUSTARC_FRAME_TEST: () => window && window.QueryString && window.QueryString.preferences === "0",
  EVAL_TRUSTARC_FRAME_GTM: () => window && window.QueryString && window.QueryString.gtm === "1",
  // declarative rules
  EVAL_ABC_TEST: () => document.cookie.includes("trackingconsent"),
  EVAL_ADROLL_0: () => !document.cookie.includes("__adroll_fpc"),
  EVAL_ALMACMP_0: () => document.cookie.includes('"name":"Google","consent":false'),
  EVAL_AFFINITY_SERIF_COM_0: () => document.cookie.includes("serif_manage_cookies_viewed") && !document.cookie.includes("serif_allow_analytics"),
  EVAL_ARBEITSAGENTUR_TEST: () => document.cookie.includes("cookie_consent=denied"),
  EVAL_AXEPTIO_0: () => document.cookie.includes("axeptio_authorized_vendors=%2C%2C"),
  EVAL_BAHN_TEST: () => utag.gdpr.getSelectedCategories().length === 1,
  EVAL_BING_0: () => document.cookie.includes("AL=0") && document.cookie.includes("AD=0") && document.cookie.includes("SM=0"),
  EVAL_BLOCKSY_0: () => document.cookie.includes("blocksy_cookies_consent_accepted=no"),
  EVAL_BORLABS_0: () => !JSON.parse(decodeURIComponent(document.cookie.split(";").find((c) => c.indexOf("borlabs-cookie") !== -1).split("=", 2)[1])).consents.statistics,
  EVAL_BUNDESREGIERUNG_DE_0: () => document.cookie.match("cookie-allow-tracking=0"),
  EVAL_CANVA_0: () => !document.cookie.includes("gtm_fpc_engagement_event"),
  EVAL_CC_BANNER2_0: () => !!document.cookie.match(/sncc=[^;]+D%3Dtrue/),
  EVAL_CLICKIO_0: () => document.cookie.includes("__lxG__consent__v2_daisybit="),
  EVAL_CLINCH_0: () => document.cookie.includes("ctc_rejected=1"),
  EVAL_COOKIECONSENT2_TEST: () => document.cookie.includes("cc_cookie="),
  EVAL_COOKIECONSENT3_TEST: () => document.cookie.includes("cc_cookie="),
  EVAL_COINBASE_0: () => JSON.parse(decodeURIComponent(document.cookie.match(/cm_(eu|default)_preferences=([0-9a-zA-Z\\{\\}\\[\\]%:]*);?/)[2])).consent.length <= 1,
  EVAL_COMPLIANZ_BANNER_0: () => document.cookie.includes("cmplz_banner-status=dismissed"),
  EVAL_COOKIE_LAW_INFO_0: () => CLI.disableAllCookies() || CLI.reject_close() || true,
  EVAL_COOKIE_LAW_INFO_1: () => document.cookie.indexOf("cookielawinfo-checkbox-non-necessary=yes") === -1,
  EVAL_COOKIE_LAW_INFO_DETECT: () => !!window.CLI,
  EVAL_COOKIE_MANAGER_POPUP_0: () => JSON.parse(document.cookie.split(";").find((c) => c.trim().startsWith("CookieLevel")).split("=")[1]).social === false,
  EVAL_COOKIEALERT_0: () => document.querySelector("body").removeAttribute("style") || true,
  EVAL_COOKIEALERT_1: () => document.querySelector("body").removeAttribute("style") || true,
  EVAL_COOKIEALERT_2: () => window.CookieConsent.declined === true,
  EVAL_COOKIEFIRST_0: () => ((o) => o.performance === false && o.functional === false && o.advertising === false)(JSON.parse(decodeURIComponent(document.cookie.split(";").find((c) => c.indexOf("cookiefirst") !== -1).trim()).split("=")[1])),
  EVAL_COOKIEFIRST_1: () => document.querySelectorAll("button[data-cookiefirst-accent-color=true][role=checkbox]:not([disabled])").forEach((i) => i.getAttribute("aria-checked") == "true" && i.click()) || true,
  EVAL_COOKIEINFORMATION_0: () => CookieInformation.declineAllCategories() || true,
  EVAL_COOKIEINFORMATION_1: () => CookieInformation.submitAllCategories() || true,
  EVAL_COOKIEINFORMATION_2: () => document.cookie.includes("CookieInformationConsent="),
  EVAL_COOKIEYES_0: () => document.cookie.includes("advertisement:no"),
  EVAL_DAILYMOTION_0: () => !!document.cookie.match("dm-euconsent-v2"),
  EVAL_DNDBEYOND_TEST: () => document.cookie.includes("cookie-consent=denied"),
  EVAL_DSGVO_0: () => !document.cookie.includes("sp_dsgvo_cookie_settings"),
  EVAL_DUNELM_0: () => document.cookie.includes("cc_functional=0") && document.cookie.includes("cc_targeting=0"),
  EVAL_ETSY_0: () => document.querySelectorAll(".gdpr-overlay-body input").forEach((toggle) => {
    toggle.checked = false;
  }) || true,
  EVAL_ETSY_1: () => document.querySelector(".gdpr-overlay-view button[data-wt-overlay-close]").click() || true,
  EVAL_EU_COOKIE_COMPLIANCE_0: () => document.cookie.indexOf("cookie-agreed=2") === -1,
  EVAL_EU_COOKIE_LAW_0: () => !document.cookie.includes("euCookie"),
  EVAL_EZOIC_0: () => ezCMP.handleAcceptAllClick(),
  EVAL_EZOIC_1: () => !!document.cookie.match(/ez-consent-tcf/),
  EVAL_FIDES_DETECT_POPUP: () => window.Fides?.initialized,
  EVAL_GOOGLE_0: () => !!document.cookie.match(/SOCS=CAE/),
  EVAL_HEMA_TEST_0: () => document.cookie.includes("cookies_rejected=1"),
  EVAL_IUBENDA_0: () => document.querySelectorAll(".purposes-item input[type=checkbox]:not([disabled])").forEach((x) => {
    if (x.checked) x.click();
  }) || true,
  EVAL_IUBENDA_1: () => !!document.cookie.match(/_iub_cs-\d+=/),
  EVAL_IWINK_TEST: () => document.cookie.includes("cookie_permission_granted=no"),
  EVAL_JQUERY_COOKIEBAR_0: () => !document.cookie.includes("cookies-state=accepted"),
  EVAL_KETCH_TEST: () => document.cookie.includes("_ketch_consent_v1_"),
  EVAL_MEDIAVINE_0: () => document.querySelectorAll('[data-name="mediavine-gdpr-cmp"] input[type=checkbox]').forEach((x) => x.checked && x.click()) || true,
  EVAL_MICROSOFT_0: () => Array.from(document.querySelectorAll("div > button")).filter((el) => el.innerText.match("Reject|Ablehnen"))[0].click() || true,
  EVAL_MICROSOFT_1: () => Array.from(document.querySelectorAll("div > button")).filter((el) => el.innerText.match("Accept|Annehmen"))[0].click() || true,
  EVAL_MICROSOFT_2: () => !!document.cookie.match("MSCC|GHCC"),
  EVAL_MOOVE_0: () => document.querySelectorAll("#moove_gdpr_cookie_modal input").forEach((i) => {
    if (!i.disabled) i.checked = i.name === "moove_gdpr_strict_cookies" || i.id === "moove_gdpr_strict_cookies";
  }) || true,
  EVAL_ONENINETWO_0: () => document.cookie.includes("CC_ADVERTISING=NO") && document.cookie.includes("CC_ANALYTICS=NO"),
  EVAL_OPENAI_TEST: () => document.cookie.includes("oai-allow-ne=false"),
  EVAL_OPERA_0: () => document.cookie.includes("cookie_consent_essential=true") && !document.cookie.includes("cookie_consent_marketing=true"),
  EVAL_PAYPAL_0: () => document.cookie.includes("cookie_prefs") === true,
  EVAL_PRIMEBOX_0: () => !document.cookie.includes("cb-enabled=accepted"),
  EVAL_PUBTECH_0: () => document.cookie.includes("euconsent-v2") && (document.cookie.match(/.YAAAAAAAAAAA/) || document.cookie.match(/.aAAAAAAAAAAA/) || document.cookie.match(/.YAAACFgAAAAA/)),
  EVAL_REDDIT_0: () => document.cookie.includes("eu_cookie={%22opted%22:true%2C%22nonessential%22:false}"),
  EVAL_ROBLOX_TEST: () => document.cookie.includes("RBXcb"),
  EVAL_SIRDATA_UNBLOCK_SCROLL: () => {
    document.documentElement.classList.forEach((cls) => {
      if (cls.startsWith("sd-cmp-")) document.documentElement.classList.remove(cls);
    });
    return true;
  },
  EVAL_SNIGEL_0: () => !!document.cookie.match("snconsent"),
  EVAL_STEAMPOWERED_0: () => JSON.parse(decodeURIComponent(document.cookie.split(";").find((s) => s.trim().startsWith("cookieSettings")).split("=")[1])).preference_state === 2,
  EVAL_SVT_TEST: () => document.cookie.includes('cookie-consent-1={"optedIn":true,"functionality":false,"statistics":false}'),
  EVAL_TAKEALOT_0: () => document.body.classList.remove("freeze") || (document.body.style = "") || true,
  EVAL_TARTEAUCITRON_0: () => tarteaucitron.userInterface.respondAll(false) || true,
  EVAL_TARTEAUCITRON_1: () => tarteaucitron.userInterface.respondAll(true) || true,
  EVAL_TARTEAUCITRON_2: () => document.cookie.match(/tarteaucitron=[^;]*/)?.[0].includes("false"),
  EVAL_TAUNTON_TEST: () => document.cookie.includes("taunton_user_consent_submitted=true"),
  EVAL_TEALIUM_0: () => typeof window.utag !== "undefined" && typeof utag.gdpr === "object",
  EVAL_TEALIUM_1: () => utag.gdpr.setConsentValue(false) || true,
  EVAL_TEALIUM_DONOTSELL: () => utag.gdpr.dns?.setDnsState(false) || true,
  EVAL_TEALIUM_2: () => utag.gdpr.setConsentValue(true) || true,
  EVAL_TEALIUM_3: () => utag.gdpr.getConsentState() !== 1,
  EVAL_TEALIUM_DONOTSELL_CHECK: () => utag.gdpr.dns?.getDnsState() !== 1,
  EVAL_TESLA_TEST: () => document.cookie.includes("tsla-cookie-consent=rejected"),
  EVAL_TESTCMP_0: () => window.results.results[0] === "button_clicked",
  EVAL_TESTCMP_COSMETIC_0: () => window.results.results[0] === "banner_hidden",
  EVAL_THEFREEDICTIONARY_0: () => cmpUi.showPurposes() || cmpUi.rejectAll() || true,
  EVAL_THEFREEDICTIONARY_1: () => cmpUi.allowAll() || true,
  EVAL_THEVERGE_0: () => document.cookie.includes("_duet_gdpr_acknowledged=1"),
  EVAL_TWCC_TEST: () => document.cookie.includes("twCookieConsent="),
  EVAL_UBUNTU_COM_0: () => document.cookie.includes("_cookies_accepted=essential"),
  EVAL_UK_COOKIE_CONSENT_0: () => !document.cookie.includes("catAccCookies"),
  EVAL_USERCENTRICS_API_0: () => typeof UC_UI === "object",
  EVAL_USERCENTRICS_API_1: () => !!UC_UI.closeCMP(),
  EVAL_USERCENTRICS_API_2: () => !!UC_UI.denyAllConsents(),
  EVAL_USERCENTRICS_API_3: () => !!UC_UI.acceptAllConsents(),
  EVAL_USERCENTRICS_API_4: () => !!UC_UI.closeCMP(),
  EVAL_USERCENTRICS_API_5: () => UC_UI.areAllConsentsAccepted() === true,
  EVAL_USERCENTRICS_API_6: () => UC_UI.areAllConsentsAccepted() === false,
  EVAL_USERCENTRICS_BUTTON_0: () => JSON.parse(localStorage.getItem("usercentrics")).consents.every((c) => c.isEssential || !c.consentStatus),
  EVAL_WAITROSE_0: () => Array.from(document.querySelectorAll("label[id$=cookies-deny-label]")).forEach((e) => e.click()) || true,
  EVAL_WAITROSE_1: () => document.cookie.includes("wtr_cookies_advertising=0") && document.cookie.includes("wtr_cookies_analytics=0"),
  EVAL_WP_COOKIE_NOTICE_0: () => document.cookie.includes("wpl_viewed_cookie=no"),
  EVAL_XE_TEST: () => document.cookie.includes("xeConsentState={%22performance%22:false%2C%22marketing%22:false%2C%22compliance%22:false}"),
  EVAL_XING_0: () => document.cookie.includes("userConsent=%7B%22marketing%22%3Afalse"),
  EVAL_YOUTUBE_DESKTOP_0: () => !!document.cookie.match(/SOCS=CAE/),
  EVAL_YOUTUBE_MOBILE_0: () => !!document.cookie.match(/SOCS=CAE/)
};

export { snippets };
