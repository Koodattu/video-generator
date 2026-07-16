# Local media library

Place owned or explicitly authorized images, GIFs, and short video clips here. The Remotion asset
resolver matches English query words against relative filenames, so use descriptive names such as
`overloaded-server-warning-lights.mp4`. Unmatched files are not selected.

The files themselves are ignored by Git. By default, a file in this directory is treated as
operator-supplied authorized media. To record explicit rights, add either
`filename.ext.license.json` or `filename.license.json` with this shape:

```json
{
  "license_id": "CC-BY-4.0",
  "license_name": "Creative Commons Attribution 4.0",
  "license_url": "https://creativecommons.org/licenses/by/4.0/",
  "terms_url": "",
  "attribution_required": true,
  "attribution_text": "Creator name",
  "share_alike": false,
  "review_status": "approved",
  "review_reason": "Licensed by the creator for reuse with attribution."
}
```
