SELECT COUNT(*)
FROM comments AS c,
  postHistory AS ph,
  votes AS v,
  users AS u
WHERE v.UserId = u.Id
  AND c.UserId = u.Id
  AND ph.UserId = u.Id
  AND ph.CreationDate >= CAST('2010-07-28 09:11:34' AS timestamp)
  AND ph.CreationDate <= CAST('2014-09-06 06:51:53' AS timestamp)
  AND u.DownVotes <= 0
  AND u.UpVotes >= 0
  AND u.UpVotes <= 72;
