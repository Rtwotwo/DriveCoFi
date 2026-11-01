const dialog = document.getElementById('figure-dialog');
const dialogImage = document.getElementById('dialog-image');
const dialogCaption = document.getElementById('dialog-caption');

document.querySelectorAll('.figure-open').forEach((button) => {
  button.addEventListener('click', () => {
    dialogImage.src = button.dataset.full;
    dialogImage.alt = button.dataset.alt || '';
    dialogCaption.textContent = button.closest('figure')?.querySelector('figcaption')?.textContent?.trim() || '';
    dialog.showModal();
  });
});

document.querySelector('.dialog-close').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', (event) => {
  if (event.target === dialog) dialog.close();
});
dialog.addEventListener('close', () => {
  dialogImage.src = '';
});

const attentionImage = document.getElementById('attention-img');
const attentionOpen = document.getElementById('attention-open');
const attentionViews = {
  front: { src: './assets/images/Figure3_1_new.webp', name: 'panel 1' },
  rear: { src: './assets/images/Figure3_2_new.webp', name: 'panel 2' }
};
document.querySelectorAll('.attention-tabs button').forEach((button) => {
  button.addEventListener('click', () => {
    document.querySelectorAll('.attention-tabs button').forEach((tab) => {
      const active = tab === button;
      tab.classList.toggle('active', active);
      tab.setAttribute('aria-selected', String(active));
    });
    const view = attentionViews[button.dataset.view];
    attentionImage.src = view.src;
    attentionImage.alt = `Register attention maps comparing DrivoR and CoFiDrive, ${view.name}`;
    attentionOpen.dataset.full = view.src;
    attentionOpen.dataset.alt = attentionImage.alt;
  });
});
