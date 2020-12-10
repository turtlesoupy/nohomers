import * as _ from 'lodash';
import { AnchorHTMLAttributes } from 'react';
import 'whatwg-fetch';

interface ContentItem {
    url: string,
    key: string,
    next_item_key: string,
    next_item_url: string,
    transition_url: string
}

declare var currentContentItem: ContentItem;

window.addEventListener('load', (event) => {
    let transitionAfter = 10000;
    let image = new Image();
    image.src = currentContentItem.next_item_url;

    var payloadImage = document.getElementById("image-payload") as HTMLImageElement;
    var payloadTransitionVideo = document.getElementById("transition-video-payload") as HTMLVideoElement;
    payloadTransitionVideo.load();

    var nextButton = document.getElementById("next-button") as HTMLLinkElement;

    function assignNext() {
        payloadImage.setAttribute("src", image.src);
        payloadImage.style.display = "";
        payloadTransitionVideo.style.display = "none";

        window.fetch(`/item/${currentContentItem.next_item_key}`)
            .then(res => res.json())
            .then(json => {
                if (!json.url) {
                    console.error("Bad response!")
                    nextButton.disabled = false;
                    nextButton.classList.remove("disabled");
                    return;
                }

                var item = json as ContentItem;
                currentContentItem = item;

                image.src = item.next_item_url;                    
                payloadTransitionVideo.src = item.transition_url;
                payloadTransitionVideo.load();
                nextButton.disabled = false;
                nextButton.classList.remove("disabled");
            })
            .catch((error) => {
                console.log(error);
                nextButton.disabled = false;
                nextButton.classList.remove("disabled");
            });

    }

    function doTransition() {
        payloadImage.style.display = "none";
        payloadTransitionVideo.style.display = "";
        nextButton.disabled = true;
        nextButton.classList.add("disabled");

        if (payloadTransitionVideo.readyState != 4) {
            assignNext();
        } else {
            payloadTransitionVideo.play();
            payloadTransitionVideo.onended = (e) => {
                assignNext();
            };
        }
    }

    var allowIdleTransition = true;
    var idleInterval = setInterval(() => { 
        if (!nextButton.disabled && document.hasFocus && allowIdleTransition) {
            doTransition();
        }
    }, transitionAfter);

    nextButton.addEventListener("click", (e) => {
        e.preventDefault();
        allowIdleTransition = false;
        if (!nextButton.disabled) {
            doTransition();
        }
    });

});