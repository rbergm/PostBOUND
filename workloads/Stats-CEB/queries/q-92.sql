SELECT COUNT(*)
FROM postHistory AS ph, posts AS p, users AS u
WHERE ph.PostId = p.Id
  AND p.OwnerUserId = u.Id
  AND p.CreationDate >= CAST('2010-08-17 19:08:05' AS timestamp)
  AND p.CreationDate <= CAST('2014-08-31 06:58:12' AS timestamp)
  AND u.UpVotes >= 0
  AND u.UpVotes <= 9;
