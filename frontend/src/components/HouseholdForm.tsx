import React, { useState } from 'react';

// Compact form for the household facts the grant and affordability calculators still lack. It sends ONE plain-English
// message made only of sentences the backend parser reads (tests/test_flat_affordability.py feeds these exact sentences
// through stated_financial_inputs; keep the two in step).

type Values = {
  citizenship: 'couple' | 'pr' | 'single';
  firstTimer: 'yes' | 'no';
  age1: string;
  age2: string;
  children: 'none' | 'yes';
  youngest: string;
  income: string;
  cpf: string;
  cash: string;
  debt: string;
  worked: 'yes' | 'no';
  parents: 'none' | 'near' | 'with';
  grantOk: 'yes' | 'no';
  proxOk: 'yes' | 'no';
  loanOk: 'yes' | 'no';
  flatType: string;
  lease: string;
};

const DEFAULTS: Values = {
  citizenship: 'couple', firstTimer: 'yes', age1: '', age2: '', children: 'none', youngest: '', income: '', cpf: '',
  cash: '', debt: '0', worked: 'yes', parents: 'none', grantOk: 'yes', proxOk: 'yes', loanOk: 'yes',
  flatType: '4 ROOM', lease: '',
};
const STORE_KEY = 'hdb-household-form';

const loadValues = (): Values => {
  try {
    const raw = localStorage.getItem(STORE_KEY);
    return raw ? { ...DEFAULTS, ...JSON.parse(raw) } : DEFAULTS;
  } catch {
    return DEFAULTS;
  }
};

const money = (v: string): string => {
  const n = Math.round(Number(String(v).replace(/[$,\s]/g, '')));
  return `$${n.toLocaleString('en-US')}`;
};
const isNum = (v: string) => v.trim() !== '' && Number.isFinite(Number(String(v).replace(/[$,\s]/g, '')));

export const composeHouseholdMessage = (needed: string[], v: Values): string => {
  const has = (...f: string[]) => f.some((x) => needed.includes(x));
  const single = v.citizenship === 'single';
  const we = single ? 'I' : 'We';
  const our = single ? 'My' : 'Our';
  const out: string[] = [];
  if (has('flat_type')) out.push(`${we} want a ${v.flatType} flat.`);
  if (has('remaining_lease_years') && isNum(v.lease)) out.push(`Assume ${Number(v.lease)} year lease.`);
  if (has('citizenship_pair')) {
    out.push(v.citizenship === 'couple' ? 'We are both Singapore citizens.'
      : v.citizenship === 'pr' ? 'One of us is a Singapore citizen and the other is a PR.'
      : 'I am a single Singapore citizen.');
  }
  if (has('first_timer_status')) {
    out.push(single ? (v.firstTimer === 'yes' ? 'I am a first-time buyer.' : 'I am not a first-time buyer.')
      : (v.firstTimer === 'yes' ? 'We are first-time buyers.' : 'We are not first-time buyers.'));
  }
  if (has('youngest_buyer_age', 'oldest_buyer_age', 'average_applicant_age') && isNum(v.age1)) {
    out.push(single || !isNum(v.age2) ? `I am ${Number(v.age1)} years old.` : `We are aged ${Number(v.age1)} and ${Number(v.age2)}.`);
  }
  if (has('youngest_core_member_age')) {
    out.push(v.children === 'none' || !isNum(v.youngest) ? `${we} have no children.`
      : `The youngest person who will live in the flat is ${Number(v.youngest)} years old.`);
  }
  if (has('monthly_income') && isNum(v.income)) out.push(`${our} household income is ${money(v.income)} a month.`);
  if (has('cpf_oa_balance') && isNum(v.cpf)) out.push(`${our} CPF OA balance is ${money(v.cpf)}.`);
  if (has('cash_savings') && isNum(v.cash)) out.push(`${we} have ${money(v.cash)} in cash savings.`);
  if (has('monthly_debt_obligations') && isNum(v.debt)) {
    out.push(Number(v.debt) === 0 ? `${we} have no monthly debt.` : `${our} monthly debt repayments are ${money(v.debt)}.`);
  }
  if (has('continuous_employment_12_months')) {
    out.push(v.worked === 'yes'
      ? (single ? 'I have worked continuously for the past 12 months.' : 'We have both worked continuously for the past 12 months.')
      : (single ? 'I have not worked continuously for the past 12 months.' : 'We have not worked continuously for the past 12 months.'));
  }
  if (has('proximity')) {
    out.push(v.parents === 'near' ? `I am buying within 4 km of my parents.`.replace('I am', single ? 'I am' : 'We are').replace('my parents', single ? 'my parents' : 'our parents')
      : v.parents === 'with' ? `${our} parents will live with ${single ? 'me' : 'us'}.`
      : `${we} ${single ? 'am' : 'are'} not buying near ${single ? 'my' : 'our'} parents.`);
  }
  if (has('eligibility_confirmed')) out.push(v.grantOk === 'yes' ? `${our} grant eligibility is confirmed.` : `${our} grant eligibility is not confirmed.`);
  // buying near or with parents: the grant calculator then needs this too, so it is sent with the same answer
  if (has('proximity_eligibility_confirmed') || (has('proximity') && v.parents !== 'none')) {
    out.push(v.proxOk === 'yes' ? `${our} proximity grant eligibility is confirmed.` : `${our} proximity grant eligibility is not confirmed.`);
  }
  if (has('loan_eligibility_confirmed')) {
    out.push(v.loanOk === 'yes' ? `${our} HDB loan eligibility is confirmed.` : `${our} HDB loan eligibility is not confirmed.`);
  }
  return out.join(' ');
};

interface Props {
  needed: string[];
  disabled?: boolean;
  onSubmit: (message: string) => void;
}

export const HouseholdForm: React.FC<Props> = ({ needed, disabled, onSubmit }) => {
  const [v, setV] = useState<Values>(loadValues);
  const has = (...f: string[]) => f.some((x) => needed.includes(x));
  const set = <K extends keyof Values>(k: K, val: Values[K]) => setV((p) => ({ ...p, [k]: val }));
  const single = v.citizenship === 'single';

  const yesNo = (k: 'firstTimer' | 'worked' | 'grantOk' | 'proxOk' | 'loanOk', label: string) => (
    <label className="hf-field">
      <span>{label}</span>
      <select value={v[k]} onChange={(e) => set(k, e.target.value as 'yes' | 'no')}>
        <option value="yes">Yes</option>
        <option value="no">No</option>
      </select>
    </label>
  );
  const num = (k: 'income' | 'cpf' | 'cash' | 'debt', label: string) => (
    <label className="hf-field">
      <span>{label}</span>
      <input type="text" inputMode="numeric" placeholder="$" value={v[k]} onChange={(e) => set(k, e.target.value)} />
    </label>
  );

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    try { localStorage.setItem(STORE_KEY, JSON.stringify(v)); } catch { /* storage unavailable */ }
    const message = composeHouseholdMessage(needed, v);
    if (message) onSubmit(message);
  };

  return (
    <form className="household-form" onSubmit={submit}>
      <div className="hf-title">Your household details</div>
      <div className="hf-grid">
        {has('citizenship_pair') && (
          <label className="hf-field">
            <span>Citizenship</span>
            <select value={v.citizenship} onChange={(e) => set('citizenship', e.target.value as Values['citizenship'])}>
              <option value="couple">Singapore citizen couple</option>
              <option value="pr">Citizen + PR</option>
              <option value="single">Single citizen</option>
            </select>
          </label>
        )}
        {has('first_timer_status') && yesNo('firstTimer', 'First-time buyers?')}
        {has('youngest_buyer_age', 'oldest_buyer_age', 'average_applicant_age') && (
          <>
            <label className="hf-field"><span>{single ? 'Your age' : 'Age of buyer 1'}</span>
              <input type="text" inputMode="numeric" value={v.age1} onChange={(e) => set('age1', e.target.value)} /></label>
            {!single && (
              <label className="hf-field"><span>Age of buyer 2</span>
                <input type="text" inputMode="numeric" value={v.age2} onChange={(e) => set('age2', e.target.value)} /></label>
            )}
          </>
        )}
        {has('youngest_core_member_age') && (
          <>
            <label className="hf-field"><span>Children living with you?</span>
              <select value={v.children} onChange={(e) => set('children', e.target.value as 'none' | 'yes')}>
                <option value="none">No children</option>
                <option value="yes">Yes</option>
              </select></label>
            {v.children === 'yes' && (
              <label className="hf-field"><span>Youngest occupant age</span>
                <input type="text" inputMode="numeric" value={v.youngest} onChange={(e) => set('youngest', e.target.value)} /></label>
            )}
          </>
        )}
        {has('monthly_income') && num('income', 'Monthly household income')}
        {has('cpf_oa_balance') && num('cpf', 'CPF OA balance')}
        {has('cash_savings') && num('cash', 'Cash savings')}
        {has('monthly_debt_obligations') && num('debt', 'Monthly debt repayments (0 if none)')}
        {has('continuous_employment_12_months') && yesNo('worked', 'Worked continuously 12 months?')}
        {has('proximity') && (
          <label className="hf-field"><span>Parents</span>
            <select value={v.parents} onChange={(e) => set('parents', e.target.value as Values['parents'])}>
              <option value="none">Not buying near parents</option>
              <option value="near">Within 4 km of parents</option>
              <option value="with">Living with parents</option>
            </select></label>
        )}
        {has('eligibility_confirmed') && yesNo('grantOk', 'Grant eligibility confirmed?')}
        {(has('proximity_eligibility_confirmed') || (has('proximity') && v.parents !== 'none')) &&
          yesNo('proxOk', 'Proximity grant confirmed?')}
        {has('flat_type') && (
          <label className="hf-field"><span>Flat type</span>
            <select value={v.flatType} onChange={(e) => set('flatType', e.target.value)}>
              {['2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE'].map((t) => <option key={t} value={t}>{t}</option>)}
            </select></label>
        )}
        {has('remaining_lease_years') && (
          <label className="hf-field"><span>Remaining lease (years)</span>
            <input type="text" inputMode="numeric" value={v.lease} onChange={(e) => set('lease', e.target.value)} /></label>
        )}
        {has('loan_eligibility_confirmed') && yesNo('loanOk', 'HDB loan eligibility confirmed?')}
      </div>
      <button type="submit" className="hf-submit" disabled={disabled}>Send these details</button>
    </form>
  );
};
