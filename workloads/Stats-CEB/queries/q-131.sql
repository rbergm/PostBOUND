SELECT COUNT(*)
FROM comments AS c,
  posts AS p,
  postHistory AS ph,
  votes AS v,
  users AS u
WHERE v.UserId = u.Id
  AND c.UserId = u.Id
  AND p.OwnerUserId = u.Id
  AND ph.UserId = u.Id
  AND c.Score = 2
  AND p.AnswerCount >= 0
  AND p.AnswerCount <= 9
  AND p.CreationDate >= CAST('2010-07-20 18:17:25' AS timestamp)
  AND p.CreationDate <= CAST('2014-08-26 12:57:22' AS timestamp)
  AND ph.CreationDate <= CAST('2014-09-02 07:58:47' AS timestamp)
  AND v.BountyAmount >= 0
  AND v.CreationDate >= CAST('2010-05-19 00:00:00' AS timestamp)
  AND u.UpVotes <= 230
  AND u.CreationDate >= CAST('2010-09-22 01:07:10' AS timestamp)
  AND u.CreationDate <= CAST('2014-08-15 05:52:23' AS timestamp);
