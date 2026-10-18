"""Generate the Rudy light and dark themes from Telegram's built-in ones.

Run from the repository root:

    python3 Tools/rudy-theme/make_theme.py

It reads Telegram's default colors (ThemeColors.java) and the original theme
files kept in Tools/rudy-theme/base/, runs every color through a port of
Theme.changeColorAccent with a monochrome accent, applies Rudy's own colors
(black header, black outgoing bubbles in light mode, charcoal in dark mode),
and writes TMessagesProj/src/main/assets/{bluebubbles,darkblue}.attheme.
The app then shows these colors as the default day and night themes.
"""
import colorsys
import os
import re

ROOT = os.getcwd()
HERE = os.path.dirname(os.path.abspath(__file__))
THEME_COLORS = os.path.join(ROOT, 'TMessagesProj/src/main/java/org/telegram/ui/ActionBar/ThemeColors.java')
THEME = os.path.join(ROOT, 'TMessagesProj/src/main/java/org/telegram/ui/ActionBar/Theme.java')
ASSETS = os.path.join(ROOT, 'TMessagesProj/src/main/assets')

CONSTANTS = {'TELEGRAM_COLOR': 0xFF229AF0, 'TELEGRAM_COLOR_TEXT': 0xFF298ACF, 'DEFAULT_BLACK_TEXT': 0xFF1A1D21,
             'Color.WHITE': 0xFFFFFFFF, 'ColorUtils.setAlphaComponent(Color.WHITE, 90)': 0x5AFFFFFF, '0': 0}

INK = 0xFF111113          # Rudy black
INK_PRESSED = 0xFF2C2C2E
HEADER_SUB = 0xFFA0A0A6
WHITE = 0xFFFFFFFF


# ---- read Telegram's color tables ----

def read_defaults():
    src = open(THEME_COLORS).read()
    names = dict(re.findall(r'colorKeysMap\.put\((key_\w+), "(\w+)"\)', src))
    defaults = {}
    for key, expr in re.findall(r'defaultColors\[(key_\w+)\] = ([^;]+);', src):
        expr = expr.strip()
        if expr in CONSTANTS:
            value = CONSTANTS[expr]
        else:
            value = int(expr, 16) if expr.lower().startswith('0x') else int(expr)
        if key in names:  # keys without a theme-file name can't be themed
            defaults[names[key]] = value & 0xFFFFFFFF
    return defaults


def key_order():
    """Key names in declaration order, plus the outgoing-message ranges."""
    src = open(THEME).read()
    names = dict(re.findall(r'colorKeysMap\.put\((key_\w+), "(\w+)"\)', open(THEME_COLORS).read()))
    order, marks = [], {}
    for m in re.finditer(r'public static final int (\w+) = colorsCount(\+\+)?;', src):
        ident, inc = m.group(1), m.group(2)
        if inc:
            order.append(names.get(ident, ident))
        else:
            marks[ident] = len(order)
    out_bubbles = set(order[marks['myMessagesBubblesStartIndex']:marks['myMessagesBubblesEndIndex']])
    out_keys = set(order[marks['myMessagesStartIndex']:marks['myMessagesEndIndex']])
    return out_bubbles, out_keys


def read_attheme(path):
    colors = {}
    for line in open(path, encoding='utf-8'):
        if '=' in line:
            k, v = line.strip().split('=', 1)
            try:
                colors[k] = int(v) & 0xFFFFFFFF
            except ValueError:
                pass
    return colors


# ---- port of Theme.changeColorAccent ----

def to_hsv(c):
    r, g, b = ((c >> 16) & 255) / 255, ((c >> 8) & 255) / 255, (c & 255) / 255
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return [h * 360, s, v]


def from_hsv(alpha, hsv):
    h, s, v = hsv
    r, g, b = colorsys.hsv_to_rgb((h % 360) / 360, max(0, min(1, s)), max(0, min(1, v)))
    return (alpha << 24) | (round(r * 255) << 16) | (round(g * 255) << 8) | round(b * 255)


def brightness(c):
    return (((c >> 16) & 255) * 0.2126 + ((c >> 8) & 255) * 0.7152 + (c & 255) * 0.0722) / 255


def change_brightness(c, amount):
    ch = [min(255, max(0, int(((c >> s) & 255) * amount))) for s in (16, 8, 0)]
    return (c & 0xFF000000) | (ch[0] << 16) | (ch[1] << 8) | ch[2]


def change_color_accent(base, accent, color, dark):
    base_hsv, accent_hsv, hsv = to_hsv(base), to_hsv(accent), to_hsv(color)
    diff_h = min(abs(hsv[0] - base_hsv[0]), abs(hsv[0] - base_hsv[0] - 360))
    if diff_h > 30:
        return color
    dist = min(1.5 * hsv[1] / base_hsv[1], 1)
    hsv[0] = hsv[0] + accent_hsv[0] - base_hsv[0]
    hsv[1] = hsv[1] * accent_hsv[1] / base_hsv[1]
    hsv[2] = hsv[2] * (1 - dist + dist * accent_hsv[2] / base_hsv[2])
    new = from_hsv(color >> 24, hsv)
    orig_b, new_b = brightness(color), brightness(new)
    if (orig_b > new_b) if dark else (orig_b < new_b):
        new = change_brightness(new, (1 - 0.6) * orig_b / max(new_b, 1e-6) + 0.6)
    return new


def gray_inverse(c):
    """Flip lightness to sit on a dark bubble, dropping hue."""
    y = 1 - brightness(c)
    g = round(y * 255)
    return (c & 0xFF000000) | (g << 16) | (g << 8) | g


# ---- build ----

def build(base_file, base_accent, accent, dark, overrides, out_bubble_text):
    defaults = read_defaults()
    out_bubbles, out_keys = key_order()
    theme = read_attheme(os.path.join(HERE, 'base', base_file))
    effective = dict(defaults)
    effective.update(theme)
    result = {}
    for key, color in effective.items():
        if key.startswith('avatar_background') or key == 'wallpaperFileOffset':
            result[key] = color  # per-user avatar colors stay as they are
            continue
        result[key] = change_color_accent(base_accent, accent, color, dark) if color else color
    if out_bubble_text:
        text, sub, seek, meta = out_bubble_text
        white_text = ['chat_messageTextOut', 'chat_messageLinkOut', 'chat_outForwardedNameText', 'chat_outViaBotNameText',
                      'chat_outReplyLine', 'chat_outReplyLine2', 'chat_outReplyNameText', 'chat_outPreviewLine',
                      'chat_outSiteNameText', 'chat_outInstant', 'chat_outInstantSelected', 'chat_outPreviewInstantText',
                      'chat_outAudioTitleText', 'chat_outFileNameText', 'chat_outContactNameText',
                      'chat_outAudioPerformerText', 'chat_outAudioPerformerSelectedText', 'chat_outMenu',
                      'chat_outMenuSelected', 'chat_outReplyMessageText', 'chat_outReplyMediaMessageText',
                      'chat_outReplyMediaMessageSelectedText', 'chat_outAudioSeekbarFill', 'chat_outVoiceSeekbarFill',
                      'chat_outLoader', 'chat_outLoaderSelected']
        meta_keys = ['chat_outTimeText', 'chat_outTimeSelectedText', 'chat_outSentCheck', 'chat_outSentCheckSelected',
                     'chat_outSentCheckRead', 'chat_outSentCheckReadSelected', 'chat_outSentClock',
                     'chat_outSentClockSelected', 'chat_outViews', 'chat_outViewsSelected']
        sub_keys = ['chat_outAudioDurationText', 'chat_outAudioDurationSelectedText', 'chat_outContactPhoneText',
                    'chat_outContactPhoneSelectedText', 'chat_outFileInfoText', 'chat_outFileInfoSelectedText',
                    'chat_outVenueInfoText', 'chat_outVenueInfoSelectedText']
        seek_keys = ['chat_outAudioProgress', 'chat_outAudioSelectedProgress', 'chat_outAudioSeekbar',
                     'chat_outAudioCacheSeekbar', 'chat_outAudioSeekbarSelected', 'chat_outVoiceSeekbar',
                     'chat_outVoiceSeekbarSelected']
        for key in out_keys - out_bubbles:
            if key in result:
                result[key] = gray_inverse(result[key])
        for k in white_text:
            result[k] = text
        for k in meta_keys:
            result[k] = meta
        for k in sub_keys:
            result[k] = sub
        for k in seek_keys:
            result[k] = seek
        for k in ('chat_outFileProgress', 'chat_outFileProgressSelected', 'chat_outMediaIcon', 'chat_outMediaIconSelected'):
            result[k] = overrides['chat_outBubble']
    result.update(overrides)
    unknown = [k for k in overrides if k not in defaults and k not in theme]
    if unknown:
        raise SystemExit(f'unknown color keys: {unknown}')
    lines = [f'{k}={v - (1 << 32) if v >= 1 << 31 else v}' for k, v in sorted(result.items())
             if v != defaults.get(k) or k in theme]
    with open(os.path.join(ASSETS, base_file), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    return result


LIGHT = {
    'actionBarDefault': INK, 'actionBarDefaultIcon': WHITE, 'actionBarDefaultTitle': WHITE,
    'actionBarDefaultSubtitle': HEADER_SUB, 'actionBarDefaultSelector': 0x2EFFFFFF,
    'actionBarDefaultSearch': WHITE, 'actionBarDefaultSearchPlaceholder': 0x88FFFFFF,
    'actionBarTabActiveText': WHITE, 'actionBarTabUnactiveText': 0xFF8E8E93, 'actionBarTabLine': WHITE,
    'actionBarTabSelector': 0x2EFFFFFF, 'telegram_color_dialogsLogo': WHITE,
    'avatar_backgroundActionBarBlue': INK, 'avatar_actionBarIconBlue': WHITE, 'avatar_actionBarSelectorBlue': 0x2EFFFFFF,
    'avatar_subtitleInProfileBlue': HEADER_SUB, 'profile_title': WHITE,
    'chats_actionBackground': INK, 'chats_actionPressedBackground': INK_PRESSED, 'chats_actionIcon': WHITE,
    'chat_outBubble': INK, 'chat_outBubbleSelected': 0xFF2A2A2D,
    'chat_messageLinkIn': 0xFF4A4A50,
    'chat_wallpaper': 0xFFE4E4E0,
}

DARK = {
    'chat_outBubble': 0xFF3A3A3F, 'chat_outBubbleSelected': 0xFF47474D,
    'telegram_color_dialogsLogo': WHITE,
    'chat_wallpaper': 0xFF141416,
}

if __name__ == '__main__':
    light = build('bluebubbles.attheme', 0xFF328ACF, 0xFF1C1C1E, False, LIGHT,
                  (WHITE, 0xFFD2D2D6, 0x4DFFFFFF, 0xFFA5A5AB))
    dark = build('darkblue.attheme', 0xFF3685FA, 0xFF8E8E93, True, DARK, None)
    for name, res in (('light', light), ('dark', dark)):
        sample = ['actionBarDefault', 'windowBackgroundWhite', 'windowBackgroundWhiteBlueText', 'chats_actionBackground',
                  'chat_inBubble', 'chat_outBubble', 'chat_messageTextOut', 'chat_outTimeText', 'chat_messagePanelSend',
                  'featuredStickers_addButton', 'featuredStickers_buttonText', 'switchTrackChecked', 'chats_unreadCounter',
                  'chats_menuTopBackgroundCats', 'chat_messageLinkIn', 'windowBackgroundWhiteGrayText2']
        print(name, ' '.join(f'{k}=#{res.get(k, 0):08X}' for k in sample if k in res))
