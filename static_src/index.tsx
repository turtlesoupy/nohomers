import * as _ from 'lodash';
import { AnchorHTMLAttributes } from 'react';
import 'whatwg-fetch';

interface ContentItem {
    url: string,
    key: string,
    next_item_key: string,
    next_item_url: string,
    transition_url: string,
    permalink: string,
}

declare var currentContentItem: ContentItem;

window.addEventListener('load', (event) => {
    let transitionAfter = 7500;
    let image = new Image();
    image.src = currentContentItem.next_item_url;

    var payloadImage = document.getElementById("image-payload") as HTMLImageElement;
    var payloadTransitionVideo = document.getElementById("transition-video-payload") as HTMLVideoElement;
    payloadTransitionVideo.load();

    var nextButton = document.getElementById("next-button") as HTMLLinkElement;
    var imageLink = document.getElementById("display-link") as HTMLLinkElement;
    var copyElement = document.getElementById("copy-a") as HTMLLinkElement;
    var twitterElement = document.getElementById("tweet-a") as HTMLLinkElement;

    copyElement.setAttribute("href", currentContentItem.permalink);

    function syncToContentItem(ci: ContentItem) {
        currentContentItem = ci;
        payloadImage.setAttribute("src", ci.url);
        image.src = ci.next_item_url;                    
        payloadTransitionVideo.src = ci.transition_url;
        payloadTransitionVideo.load();
        copyElement.setAttribute("href", ci.permalink);
        twitterElement.setAttribute("href",
            `https://twitter.com/intent/tweet?url=${encodeURIComponent(copyElement.href)}`
        );
    }

    function assignNext() {
        payloadImage.setAttribute("src", image.src);
        payloadImage.style.zIndex = "1";
        payloadTransitionVideo.style.zIndex = "0";

        setTimeout(function() {
            window.fetch(`/item/${currentContentItem.next_item_key}`)
                .then(res => res.json())
                .then(json => {
                    if (!json.url) {
                        console.error("Bad response!")
                        nextButton.disabled = false;
                        nextButton.classList.remove("disabled");
                        return;
                    }

                    history.pushState({
                        contentItem: currentContentItem
                    }, "", currentContentItem.permalink);
                    syncToContentItem(json as ContentItem);

                    nextButton.disabled = false;
                    nextButton.classList.remove("disabled");
                })
                .catch((error) => {
                    console.log(error);
                    nextButton.disabled = false;
                    nextButton.classList.remove("disabled");
                });
        }, 50);

    }

    function doTransition() {
        payloadImage.style.zIndex = "0";
        payloadTransitionVideo.style.zIndex = "1";
        nextButton.disabled = true;
        nextButton.classList.add("disabled");

        function handleTransition() {
            if (payloadTransitionVideo.readyState != 4) {
                assignNext();
            } else {
                payloadTransitionVideo.play();
                payloadTransitionVideo.onended = (e) => {
                    assignNext();
                };
            }
        }

        if (payloadTransitionVideo.readyState != 4) {
            setTimeout(handleTransition, 500);
        } else {
            handleTransition();
        }
    }

    var allowIdleTransition = true;
    var idleInterval = setInterval(() => { 
        if (!nextButton.disabled && document.hasFocus && allowIdleTransition) {
            doTransition();
        }
    }, transitionAfter);

    
    function userInitiatedTransition() {
        allowIdleTransition = false;
        clearInterval(idleInterval);
        if (!nextButton.disabled) {
            doTransition();
        }
    }

    nextButton.addEventListener("click", (e) => {
        e.preventDefault();
        userInitiatedTransition();
    });
    imageLink.addEventListener("click", (e) => {
        e.preventDefault();
        userInitiatedTransition();
    });

    copyElement.addEventListener("click", (e) => {
        e.preventDefault();
        var copyText = copyElement.href;

        document.addEventListener('copy', function(e) {
            e.clipboardData.setData('text/plain', copyText);
            e.preventDefault();
        }, true);

        document.execCommand('copy');  
    });

    window.addEventListener("popstate", (e) => {
        console.log(e.state.contentItem);
        syncToContentItem(e.state.contentItem);
    });
});