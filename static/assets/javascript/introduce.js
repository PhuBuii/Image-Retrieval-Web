const closeBtn = document.querySelector("#close");
const modal = document.querySelector(".js-modal");
const page = document.querySelector("container-page");

function closeModal() {
  modal.classList.remove("open");
}
closeBtn.addEventListener("click", closeModal);
