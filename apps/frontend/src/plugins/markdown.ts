import MarkdownIt from 'markdown-it';
import mdColorDefault from 'markdown-it-color';
// @ts-expect-error - no types for this package
import mdTaskListsDefault from 'markdown-it-deflist';

import { defineNuxtPlugin } from '#app';

// Helper to get the actual function, handling CJS/ESM differences
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const unwrapDefault = (mod: any) => mod.default || mod;

const markdownItColor = unwrapDefault(mdColorDefault);
const markdownItTaskLists = unwrapDefault(mdTaskListsDefault);

export default defineNuxtPlugin({
  name: 'markdown-it',
  setup() {
    const md = new MarkdownIt({
      linkify: true,
      breaks: true,
      typographer: true,
      html: true, // Be careful with this if markdown comes from users!
    })
      .use(markdownItTaskLists)
      .use(markdownItColor, { defaultClassName: 'text-primary' });

    return {
      provide: {
        md: md, // Provide the configured instance
      },
    };
  },
});
