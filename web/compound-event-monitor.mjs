import { CompoundEventType } from './compound-runtime.mjs';

export const GM_PROGRAM_NAMES = Object.freeze([
  'Acoustic Grand Piano','Bright Acoustic Piano','Electric Grand Piano','Honky-tonk Piano','Electric Piano 1','Electric Piano 2','Harpsichord','Clavinet',
  'Celesta','Glockenspiel','Music Box','Vibraphone','Marimba','Xylophone','Tubular Bells','Dulcimer',
  'Drawbar Organ','Percussive Organ','Rock Organ','Church Organ','Reed Organ','Accordion','Harmonica','Tango Accordion',
  'Acoustic Guitar (nylon)','Acoustic Guitar (steel)','Electric Guitar (jazz)','Electric Guitar (clean)','Electric Guitar (muted)','Overdriven Guitar','Distortion Guitar','Guitar Harmonics',
  'Acoustic Bass','Electric Bass (finger)','Electric Bass (pick)','Fretless Bass','Slap Bass 1','Slap Bass 2','Synth Bass 1','Synth Bass 2',
  'Violin','Viola','Cello','Contrabass','Tremolo Strings','Pizzicato Strings','Orchestral Harp','Timpani',
  'String Ensemble 1','String Ensemble 2','Synth Strings 1','Synth Strings 2','Choir Aahs','Voice Oohs','Synth Voice','Orchestra Hit',
  'Trumpet','Trombone','Tuba','Muted Trumpet','French Horn','Brass Section','Synth Brass 1','Synth Brass 2',
  'Soprano Sax','Alto Sax','Tenor Sax','Baritone Sax','Oboe','English Horn','Bassoon','Clarinet',
  'Piccolo','Flute','Recorder','Pan Flute','Blown Bottle','Shakuhachi','Whistle','Ocarina',
  'Lead 1 (square)','Lead 2 (sawtooth)','Lead 3 (calliope)','Lead 4 (chiff)','Lead 5 (charang)','Lead 6 (voice)','Lead 7 (fifths)','Lead 8 (bass + lead)',
  'Pad 1 (new age)','Pad 2 (warm)','Pad 3 (polysynth)','Pad 4 (choir)','Pad 5 (bowed)','Pad 6 (metallic)','Pad 7 (halo)','Pad 8 (sweep)',
  'FX 1 (rain)','FX 2 (soundtrack)','FX 3 (crystal)','FX 4 (atmosphere)','FX 5 (brightness)','FX 6 (goblins)','FX 7 (echoes)','FX 8 (sci-fi)',
  'Sitar','Banjo','Shamisen','Koto','Kalimba','Bag Pipe','Fiddle','Shanai',
  'Tinkle Bell','Agogo','Steel Drums','Woodblock','Taiko Drum','Melodic Tom','Synth Drum','Reverse Cymbal',
  'Guitar Fret Noise','Breath Noise','Seashore','Bird Tweet','Telephone Ring','Helicopter','Applause','Gunshot',
]);

export const EVENT_TYPE_NAMES = Object.freeze({
  [CompoundEventType.NOTE]: 'NOTE',
  [CompoundEventType.CC]: 'CC',
  [CompoundEventType.PROGRAM]: 'PROGRAM',
  [CompoundEventType.BANK]: 'BANK',
  [CompoundEventType.TEMPO]: 'TEMPO',
  [CompoundEventType.PEDAL]: 'PEDAL',
  [CompoundEventType.PITCH_BEND]: 'PITCH_BEND',
  [CompoundEventType.CHANNEL_PRESSURE]: 'CHANNEL_PRESSURE',
  [CompoundEventType.POLY_PRESSURE]: 'POLY_PRESSURE',
  [CompoundEventType.TIME_SIGNATURE]: 'TIME_SIGNATURE',
});

function clampProgram(value) { return Math.max(0, Math.min(127, Number(value) || 0)); }

export function createMonitorState() {
  return {
    channels: Array.from({ length: 16 }, (_, channel) => ({ channel, program: 0, bankMsb: 0, bankLsb: 0, notes: 0, programExplicit: false })),
    eventCounts: Object.fromEntries(Object.values(EVENT_TYPE_NAMES).map((name) => [name, 0])),
    recent: [],
    totalEvents: 0,
  };
}

export function describeCompoundEvent(event, state) {
  const typeName = EVENT_TYPE_NAMES[event.type] || `TYPE_${event.type}`;
  const channel = Math.max(0, Math.min(15, Number(event.channel) || 0));
  const voice = state.channels[channel];
  if (event.type === CompoundEventType.PROGRAM) {
    return `PROGRAM · ch ${channel + 1} · ${clampProgram(event.a1)} ${GM_PROGRAM_NAMES[clampProgram(event.a1)]}`;
  }
  if (event.type === CompoundEventType.BANK) return `BANK · ch ${channel + 1} · MSB ${event.a1} · LSB ${event.a2}`;
  if (event.type === CompoundEventType.NOTE) {
    const instrument = channel === 9 ? 'Drum kit' : `${voice.program} ${GM_PROGRAM_NAMES[voice.program]}`;
    return `NOTE · ch ${channel + 1} · pitch ${event.a1} · vel ${event.a3} · ${instrument}`;
  }
  if (event.type === CompoundEventType.CC) return `CC · ch ${channel + 1} · #${event.a1}=${event.a2}`;
  if (event.type === CompoundEventType.TEMPO) return `TEMPO · ${event.a1} BPM`;
  if (event.type === CompoundEventType.PEDAL) return `PEDAL · ch ${channel + 1} · ${event.a1 ? 'down' : 'up'}`;
  return `${typeName} · ch ${channel + 1}`;
}

export function consumeMonitorEvents(state, events, { recentLimit = 36 } = {}) {
  for (const event of events) {
    const channel = Math.max(0, Math.min(15, Number(event.channel) || 0));
    const voice = state.channels[channel];
    const description = describeCompoundEvent(event, state);
    const typeName = EVENT_TYPE_NAMES[event.type] || `TYPE_${event.type}`;
    state.eventCounts[typeName] = (state.eventCounts[typeName] || 0) + 1;
    state.totalEvents += 1;
    if (event.type === CompoundEventType.PROGRAM) {
      voice.program = clampProgram(event.a1);
      voice.programExplicit = true;
    } else if (event.type === CompoundEventType.BANK) {
      voice.bankMsb = Number(event.a1) || 0;
      voice.bankLsb = Number(event.a2) || 0;
    } else if (event.type === CompoundEventType.NOTE) {
      voice.notes += 1;
    }
    state.recent.push({ type: typeName, channel, description });
    if (state.recent.length > recentLimit) state.recent.splice(0, state.recent.length - recentLimit);
  }
  return state;
}

export function monitorSnapshot(state) {
  const activeChannels = state.channels.filter((channel) => channel.notes > 0 || channel.programExplicit);
  return {
    totalEvents: state.totalEvents,
    eventCounts: { ...state.eventCounts },
    channels: activeChannels.map((channel) => ({
      ...channel,
      instrument: channel.channel === 9 ? 'Drum kit' : GM_PROGRAM_NAMES[channel.program],
    })),
    recent: state.recent.slice(),
  };
}

export class CompoundEventMonitor {
  constructor({ channelsElement = null, recentElement = null, summaryElement = null } = {}) {
    this.channelsElement = channelsElement;
    this.recentElement = recentElement;
    this.summaryElement = summaryElement;
    this.state = createMonitorState();
  }
  reset() { this.state = createMonitorState(); this.render(); }
  consume(events) { consumeMonitorEvents(this.state, events); this.render(); return monitorSnapshot(this.state); }
  snapshot() { return monitorSnapshot(this.state); }
  render() {
    const snapshot = this.snapshot();
    if (this.summaryElement) {
      const c = snapshot.eventCounts;
      this.summaryElement.textContent = `events ${snapshot.totalEvents} · NOTE ${c.NOTE || 0} · PROGRAM ${c.PROGRAM || 0} · BANK ${c.BANK || 0} · CC ${c.CC || 0}`;
    }
    if (this.channelsElement) {
      this.channelsElement.replaceChildren();
      const channels = snapshot.channels.length ? snapshot.channels : [{ channel: 0, program: 0, instrument: 'No audible channel yet', notes: 0, programExplicit: false }];
      for (const channel of channels) {
        const row = document.createElement('div');
        row.className = 'event-monitor-channel';
        const label = channel.instrument === 'No audible channel yet'
          ? channel.instrument
          : `Ch ${channel.channel + 1} · ${channel.channel === 9 ? channel.instrument : `${channel.program} ${channel.instrument}`} · notes ${channel.notes}`;
        row.textContent = label;
        this.channelsElement.appendChild(row);
      }
    }
    if (this.recentElement) {
      this.recentElement.replaceChildren();
      for (const item of snapshot.recent.slice().reverse()) {
        const row = document.createElement('div');
        row.className = `event-monitor-event event-${item.type.toLowerCase().replaceAll('_', '-')}`;
        row.textContent = item.description;
        this.recentElement.appendChild(row);
      }
    }
  }
}
