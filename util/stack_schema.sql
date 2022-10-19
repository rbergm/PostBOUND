CREATE TABLE account (
    id integer NOT NULL,
    display_name character varying,
    location character varying,
    about_me character varying,
    website_url character varying
);

CREATE TABLE answer (
    id integer NOT NULL,
    site_id integer NOT NULL,
    question_id integer,
    creation_date timestamp without time zone,
    deletion_date timestamp without time zone,
    score integer,
    view_count integer,
    body character varying,
    owner_user_id integer,
    last_editor_id integer,
    last_edit_date timestamp without time zone,
    last_activity_date timestamp without time zone,
    title character varying
);

CREATE TABLE badge (
    site_id integer NOT NULL,
    user_id integer NOT NULL,
    name character varying NOT NULL,
    date timestamp without time zone NOT NULL
);


CREATE TABLE comment (
    id integer NOT NULL,
    site_id integer NOT NULL,
    post_id integer,
    user_id integer,
    score integer,
    body character varying,
    date timestamp without time zone
);

CREATE TABLE post_link (
    site_id integer NOT NULL,
    post_id_from integer NOT NULL,
    post_id_to integer NOT NULL,
    link_type integer NOT NULL,
    date timestamp without time zone
);

CREATE TABLE question (
    id integer NOT NULL,
    site_id integer NOT NULL,
    accepted_answer_id integer,
    creation_date timestamp without time zone,
    deletion_date timestamp without time zone,
    score integer,
    view_count integer,
    body character varying,
    owner_user_id integer,
    last_editor_id integer,
    last_edit_date timestamp without time zone,
    last_activity_date timestamp without time zone,
    title character varying,
    favorite_count integer,
    closed_date timestamp without time zone,
    tagstring character varying
);


CREATE TABLE site (
    site_id integer NOT NULL,
    site_name character varying
);

CREATE TABLE so_user (
    id integer NOT NULL,
    site_id integer NOT NULL,
    reputation integer,
    creation_date timestamp without time zone,
    last_access_date timestamp without time zone,
    upvotes integer,
    downvotes integer,
    account_id integer
);

CREATE TABLE tag (
    id integer NOT NULL,
    site_id integer NOT NULL,
    name character varying
);


CREATE TABLE tag_question (
    question_id integer NOT NULL,
    tag_id integer NOT NULL,
    site_id integer NOT NULL
);

ALTER TABLE ONLY account ADD CONSTRAINT account_pkey PRIMARY KEY (id);
ALTER TABLE ONLY answer ADD CONSTRAINT answer_pkey PRIMARY KEY (id, site_id);
ALTER TABLE ONLY badge ADD CONSTRAINT badge_pkey PRIMARY KEY (site_id, user_id, name, date);
ALTER TABLE ONLY comment ADD CONSTRAINT comment_pkey PRIMARY KEY (id, site_id);
ALTER TABLE ONLY post_link ADD CONSTRAINT post_link_pkey PRIMARY KEY (site_id, post_id_from, post_id_to, link_type);
ALTER TABLE ONLY question ADD CONSTRAINT question_pkey PRIMARY KEY (id, site_id);
ALTER TABLE ONLY site ADD CONSTRAINT site_pkey PRIMARY KEY (site_id);
ALTER TABLE ONLY so_user ADD CONSTRAINT so_user_pkey PRIMARY KEY (id, site_id);
ALTER TABLE ONLY tag ADD CONSTRAINT tag_pkey PRIMARY KEY (id, site_id);
ALTER TABLE ONLY tag_question ADD CONSTRAINT tag_question_pkey PRIMARY KEY (site_id, question_id, tag_id);

CREATE INDEX answer_creation_date_idx ON answer USING btree (creation_date);
CREATE INDEX answer_last_editor_id_idx ON answer USING btree (last_editor_id);
CREATE INDEX answer_owner_user_id_idx ON answer USING btree (owner_user_id);
CREATE INDEX answer_site_id_question_id_idx ON answer USING btree (site_id, question_id);
CREATE INDEX comment_site_id_post_id_idx ON comment USING btree (site_id, post_id);
CREATE INDEX comment_site_id_user_id_idx ON comment USING btree (site_id, user_id);
CREATE INDEX question_creation_date_idx ON question USING btree (creation_date);
CREATE INDEX question_last_editor_id_idx ON question USING btree (last_editor_id);
CREATE INDEX question_owner_user_id_idx ON question USING btree (owner_user_id);
CREATE INDEX so_user_creation_date_idx ON so_user USING btree (creation_date);
CREATE INDEX so_user_last_access_date_idx ON so_user USING btree (last_access_date);
CREATE INDEX tag_question_site_id_tag_id_question_id_idx ON tag_question USING btree (site_id, tag_id, question_id);

ALTER TABLE ONLY answer ADD CONSTRAINT answer_site_id_fkey FOREIGN KEY (site_id) REFERENCES site(site_id);
ALTER TABLE ONLY answer ADD CONSTRAINT answer_site_id_fkey1 FOREIGN KEY (site_id, owner_user_id) REFERENCES so_user(site_id, id);
ALTER TABLE ONLY answer ADD CONSTRAINT answer_site_id_fkey2 FOREIGN KEY (site_id, last_editor_id) REFERENCES so_user(site_id, id);
ALTER TABLE ONLY answer ADD CONSTRAINT answer_site_id_fkey3 FOREIGN KEY (site_id, question_id) REFERENCES question(site_id, id);
ALTER TABLE ONLY badge ADD CONSTRAINT badge_site_id_fkey FOREIGN KEY (site_id) REFERENCES site(site_id);
ALTER TABLE ONLY badge ADD CONSTRAINT badge_site_id_fkey1 FOREIGN KEY (site_id, user_id) REFERENCES so_user(site_id, id);
ALTER TABLE ONLY comment ADD CONSTRAINT comment_site_id_fkey FOREIGN KEY (site_id) REFERENCES site(site_id);
ALTER TABLE ONLY post_link ADD CONSTRAINT post_link_site_id_fkey FOREIGN KEY (site_id) REFERENCES site(site_id);
ALTER TABLE ONLY post_link ADD CONSTRAINT post_link_site_id_fkey1 FOREIGN KEY (site_id, post_id_to) REFERENCES question(site_id, id);
ALTER TABLE ONLY post_link ADD CONSTRAINT post_link_site_id_fkey2 FOREIGN KEY (site_id, post_id_from) REFERENCES question(site_id, id);
ALTER TABLE ONLY question ADD CONSTRAINT question_site_id_fkey FOREIGN KEY (site_id) REFERENCES site(site_id);
ALTER TABLE ONLY question ADD CONSTRAINT question_site_id_fkey1 FOREIGN KEY (site_id, owner_user_id) REFERENCES so_user(site_id, id);
ALTER TABLE ONLY question ADD CONSTRAINT question_site_id_fkey2 FOREIGN KEY (site_id, last_editor_id) REFERENCES so_user(site_id, id);
ALTER TABLE ONLY so_user ADD CONSTRAINT so_user_account_id_fkey FOREIGN KEY (account_id) REFERENCES account(id);
ALTER TABLE ONLY so_user ADD CONSTRAINT so_user_site_id_fkey FOREIGN KEY (site_id) REFERENCES site(site_id);
ALTER TABLE ONLY tag_question ADD CONSTRAINT tag_question_site_id_fkey FOREIGN KEY (site_id) REFERENCES site(site_id);
ALTER TABLE ONLY tag_question ADD CONSTRAINT tag_question_site_id_fkey1 FOREIGN KEY (site_id, tag_id) REFERENCES tag(site_id, id);
ALTER TABLE ONLY tag_question ADD CONSTRAINT tag_question_site_id_fkey2 FOREIGN KEY (site_id, question_id) REFERENCES question(site_id, id) ON DELETE CASCADE;
ALTER TABLE ONLY tag ADD CONSTRAINT tag_site_id_fkey FOREIGN KEY (site_id) REFERENCES site(site_id);
